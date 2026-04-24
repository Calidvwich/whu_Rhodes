from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .facenet_submodule import InceptionResnetV1

VECTOR_DIM = 512
ImageLike = str | Path | Image.Image | np.ndarray
DEFAULT_FACE_SIZE = (160, 160)

_EXTRACTOR_CACHE: dict[tuple[str, str | None], "FaceNetExtractor"] = {}


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n


def _resolve_device(device: str | None) -> str:
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device={device!r}, expected one of: auto/cpu/cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but CUDA is not available")
    return device


def _to_pil_rgb(image: ImageLike) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"numpy image must have shape (H, W, 3), got {arr.shape}")
        if arr.dtype in (np.float32, np.float64):
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    raise TypeError(f"unsupported image type: {type(image)!r}")


def _pil_to_facenet_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return (t - 127.5) / 128.0


class FaceNetExtractor:
    def __init__(self, *, device: str | None = None, model_cache: str | Path | None = None) -> None:
        self.device = _resolve_device(device)
        if model_cache is not None:
            os.environ["TORCH_HOME"] = Path(model_cache).expanduser().resolve().as_posix()
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def extract_from_aligned_face(
        self,
        image: ImageLike,
        *,
        l2_normalize_output: bool = True,
        expected_size: tuple[int, int] = DEFAULT_FACE_SIZE,
    ) -> np.ndarray:
        face = _to_pil_rgb(image)
        if face.size != expected_size:
            try:
                resample = Image.Resampling.BILINEAR
            except AttributeError:
                resample = Image.BILINEAR
            face = face.resize(expected_size, resample=resample)

        t = _pil_to_facenet_tensor(face).to(self.device)
        with torch.no_grad():
            vec = self.model(t).detach().cpu().numpy().astype(np.float32).reshape(-1)

        if vec.shape != (VECTOR_DIM,):
            raise RuntimeError(f"unexpected FaceNet output shape {vec.shape}, expected ({VECTOR_DIM},)")

        return l2_normalize(vec) if l2_normalize_output else vec


def get_extractor(*, device: str | None = None, model_cache: str | Path | None = None) -> FaceNetExtractor:
    resolved_device = _resolve_device(device)
    cache_key = (resolved_device, None if model_cache is None else str(Path(model_cache)))
    extractor = _EXTRACTOR_CACHE.get(cache_key)
    if extractor is None:
        extractor = FaceNetExtractor(device=resolved_device, model_cache=model_cache)
        _EXTRACTOR_CACHE[cache_key] = extractor
    return extractor


def extract_from_aligned_face(
    image: ImageLike,
    l2_normalize_output: bool = True,
    *,
    device: str | None = None,
    model_cache: str | Path | None = None,
    expected_size: tuple[int, int] = DEFAULT_FACE_SIZE,
) -> np.ndarray:
    extractor = get_extractor(device=device, model_cache=model_cache)
    return extractor.extract_from_aligned_face(
        image,
        l2_normalize_output=l2_normalize_output,
        expected_size=expected_size,
    )


def vector_to_json_obj(vector: np.ndarray) -> dict[str, Any]:
    vec = np.asarray(vector, dtype=np.float32)
    if vec.shape != (VECTOR_DIM,):
        raise ValueError(f"vector must be shape ({VECTOR_DIM},), got {vec.shape}")
    return {"vector": vec.tolist()}


def save_vector_json(vector: np.ndarray, out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = vector_to_json_obj(vector)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
