from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .facenet_extractor import ImageLike, _to_pil_rgb
from .facenet_submodule import MTCNN

DEFAULT_OUTPUT_SIZE = 160
DEFAULT_MTCNN_THRESHOLDS = (0.6, 0.7, 0.7)

# Standard 5-point facial landmark template in 160x160 space.
REFERENCE_FACIAL_POINTS_160 = np.array(
    [
        [61.4356, 54.6963],
        [98.5620, 54.6963],
        [80.0000, 76.2514],
        [64.5297, 97.7508],
        [95.4703, 97.7508],
    ],
    dtype=np.float32,
)

_PREPROCESSOR_CACHE: dict[tuple[str, int, float], "FacePreprocessor"] = {}


def _resolve_device(device: str | None) -> str:
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device={device!r}, expected one of: auto/cpu/cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but CUDA is not available")
    return device


def _scaled_reference_points(output_size: int) -> np.ndarray:
    if output_size <= 0:
        raise ValueError(f"output_size must be positive, got {output_size}")
    scale = float(output_size) / float(DEFAULT_OUTPUT_SIZE)
    return REFERENCE_FACIAL_POINTS_160 * scale


def _estimate_similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != (5, 2) or dst.shape != (5, 2):
        raise ValueError(f"expected landmark shapes (5, 2), got {src.shape} and {dst.shape}")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    cov = (dst_demean.T @ src_demean) / src.shape[0]
    u, singular_values, vt = np.linalg.svd(cov)
    sign = np.ones(2, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1.0

    rotation = u @ np.diag(sign) @ vt
    src_var = np.mean(np.sum(src_demean * src_demean, axis=1))
    if src_var <= 1e-12:
        raise ValueError("landmarks are degenerate and cannot define a similarity transform")

    scale = float(np.sum(singular_values * sign) / src_var)
    translation = dst_mean - scale * (rotation @ src_mean)

    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = scale * rotation
    matrix[:2, 2] = translation
    return matrix


def _warp_face(image: Image.Image, src_landmarks: np.ndarray, output_size: int) -> Image.Image:
    dst_landmarks = _scaled_reference_points(output_size)
    forward = _estimate_similarity_transform(src_landmarks, dst_landmarks)
    inverse = np.linalg.inv(forward)
    coeffs = inverse[:2, :].reshape(-1).tolist()
    try:
        resample = Image.Resampling.BILINEAR
        transform_method = Image.Transform.AFFINE
    except AttributeError:
        resample = Image.BILINEAR
        transform_method = Image.AFFINE
    return image.transform((output_size, output_size), transform_method, coeffs, resample=resample)


@dataclass(frozen=True)
class FaceDetection:
    box: np.ndarray
    landmarks: np.ndarray
    probability: float
    area: float


@dataclass(frozen=True)
class PreprocessResult:
    image: Image.Image
    box: np.ndarray
    landmarks: np.ndarray
    probability: float


class FacePreprocessor:
    def __init__(
        self,
        *,
        device: str | None = None,
        output_size: int = DEFAULT_OUTPUT_SIZE,
        min_confidence: float = 0.0,
        min_face_size: int = 20,
        thresholds: tuple[float, float, float] = DEFAULT_MTCNN_THRESHOLDS,
        factor: float = 0.709,
    ) -> None:
        self.device = _resolve_device(device)
        self.output_size = int(output_size)
        self.min_confidence = float(min_confidence)
        self.mtcnn = MTCNN(
            image_size=self.output_size,
            margin=0,
            min_face_size=min_face_size,
            thresholds=list(thresholds),
            factor=factor,
            post_process=False,
            select_largest=True,
            keep_all=True,
            device=self.device,
        )

    def detect_largest_face(self, image: ImageLike) -> tuple[Image.Image, FaceDetection]:
        pil_image = _to_pil_rgb(image)
        boxes, probs, points = self.mtcnn.detect(pil_image, landmarks=True)
        if boxes is None or probs is None or points is None or len(boxes) == 0:
            raise RuntimeError("no face detected in image")

        detections: list[FaceDetection] = []
        for box, prob, landmark in zip(boxes, probs, points):
            if box is None or prob is None or landmark is None:
                continue
            box_arr = np.asarray(box, dtype=np.float32).reshape(-1)
            landmark_arr = np.asarray(landmark, dtype=np.float32).reshape(5, 2)
            width = max(0.0, float(box_arr[2] - box_arr[0]))
            height = max(0.0, float(box_arr[3] - box_arr[1]))
            detections.append(
                FaceDetection(
                    box=box_arr,
                    landmarks=landmark_arr,
                    probability=float(prob),
                    area=width * height,
                )
            )

        if not detections:
            raise RuntimeError("MTCNN returned no valid face candidates")

        candidates = [d for d in detections if d.probability >= self.min_confidence]
        if not candidates:
            best = max(detections, key=lambda d: d.probability)
            raise RuntimeError(
                "face detected but confidence is below threshold: "
                f"best={best.probability:.4f}, required>={self.min_confidence:.4f}"
            )

        target = max(candidates, key=lambda d: d.area)
        return pil_image, target

    def preprocess(self, image: ImageLike) -> PreprocessResult:
        pil_image, target = self.detect_largest_face(image)
        aligned = _warp_face(pil_image, target.landmarks, self.output_size).convert("RGB")
        return PreprocessResult(
            image=aligned,
            box=target.box.copy(),
            landmarks=target.landmarks.copy(),
            probability=target.probability,
        )


def get_preprocessor(
    *,
    device: str | None = None,
    output_size: int = DEFAULT_OUTPUT_SIZE,
    min_confidence: float = 0.0,
) -> FacePreprocessor:
    resolved_device = _resolve_device(device)
    cache_key = (resolved_device, int(output_size), float(min_confidence))
    preprocessor = _PREPROCESSOR_CACHE.get(cache_key)
    if preprocessor is None:
        preprocessor = FacePreprocessor(
            device=resolved_device,
            output_size=output_size,
            min_confidence=min_confidence,
        )
        _PREPROCESSOR_CACHE[cache_key] = preprocessor
    return preprocessor


def preprocess_face_image(
    image: ImageLike,
    *,
    device: str | None = None,
    output_size: int = DEFAULT_OUTPUT_SIZE,
    min_confidence: float = 0.0,
) -> Image.Image:
    preprocessor = get_preprocessor(
        device=device,
        output_size=output_size,
        min_confidence=min_confidence,
    )
    return preprocessor.preprocess(image).image


def preprocess_face(
    image: ImageLike,
    *,
    device: str | None = None,
    output_size: int = DEFAULT_OUTPUT_SIZE,
    min_confidence: float = 0.0,
) -> PreprocessResult:
    preprocessor = get_preprocessor(
        device=device,
        output_size=output_size,
        min_confidence=min_confidence,
    )
    return preprocessor.preprocess(image)


def save_preprocessed_face(image: Image.Image, out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
