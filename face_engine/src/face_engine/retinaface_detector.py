from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .facenet_extractor import ImageLike, _to_pil_rgb
from .retinaface_torch import TorchRetinaFace, default_model_dir

# Keep a moderate detector threshold here and leave business-side confidence
# filtering to FacePreprocessor.min_confidence for API compatibility.
DEFAULT_RETINAFACE_THRESHOLD = 0.5


@dataclass(frozen=True)
class RetinaFaceRawDetection:
    box: np.ndarray
    landmarks: np.ndarray
    probability: float


class RetinaFaceDetector:
    def __init__(
        self,
        *,
        device: str | None = None,
        detection_threshold: float = DEFAULT_RETINAFACE_THRESHOLD,
        max_size: int = 640,
    ) -> None:
        self.device = "cpu" if not device or device == "auto" else device
        self.detection_threshold = float(detection_threshold)
        self.max_size = int(max_size)
        self._model = None

    def _ensure_detector(self) -> None:
        if self._model is not None:
            return

        try:
            self._model = TorchRetinaFace(
                device=self.device,
                max_size=self.max_size,
                model_dir=default_model_dir(),
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize PyTorch RetinaFace. Confirm torch/torchvision are usable and "
                "that RetinaFace weights can be downloaded or preloaded."
            ) from exc

    @staticmethod
    def _to_rgb_numpy(image: ImageLike) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(_to_pil_rgb(image), dtype=np.uint8))

    def detect(
        self,
        image: ImageLike,
        *,
        min_confidence: float = 0.0,
    ) -> list[RetinaFaceRawDetection]:
        self._ensure_detector()
        rgb = self._to_rgb_numpy(image)
        threshold = max(float(min_confidence), self.detection_threshold)

        try:
            resp = self._model.predict(
                rgb,
                confidence_threshold=threshold,
            )
        except Exception as exc:
            raise RuntimeError(
                "PyTorch RetinaFace detection failed. Confirm the model weights are available and "
                "the current torch/torchvision runtime is healthy."
            ) from exc

        if not resp:
            return []

        detections: list[RetinaFaceRawDetection] = []
        for item in resp:
            detections.append(
                RetinaFaceRawDetection(
                    box=np.asarray(item.bbox, dtype=np.float32).reshape(4),
                    landmarks=np.asarray(item.landmarks, dtype=np.float32).reshape(5, 2),
                    probability=float(item.score),
                )
            )

        return detections
