from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from urllib.request import urlretrieve
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import _utils
from torchvision.ops import nms

MODEL_NAME = "resnet50_2020-07-20"
MODEL_URL = (
    "https://github.com/ternaus/retinaface/releases/download/0.01/"
    "retinaface_resnet50_2020-07-20-f168fae3c.zip"
)
MODEL_ARCHIVE_NAME = "retinaface_resnet50_2020-07-20-f168fae3c.zip"
MODEL_FILE_NAME = "retinaface_resnet50_2020-07-20.pth"
DEFAULT_MAX_SIZE = 640
DEFAULT_NMS_THRESHOLD = 0.4
VARIANCE = (0.1, 0.2)
MIN_SIZES = [[16, 32], [64, 128], [256, 512]]
STEPS = [8, 16, 32]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_model_dir() -> Path:
    return _repo_root() / "face_engine" / ".model_cache" / "retinaface_torch"


def _download_model_archive(model_dir: Path) -> Path:
    archive_path = model_dir / MODEL_ARCHIVE_NAME
    if archive_path.is_file():
        return archive_path
    try:
        urlretrieve(MODEL_URL, archive_path.as_posix())
    except Exception as exc:
        raise RuntimeError(
            "Failed to download RetinaFace PyTorch weights archive from the configured release URL."
        ) from exc
    return archive_path


def _ensure_model_file(model_dir: Path) -> Path:
    model_path = model_dir / MODEL_FILE_NAME
    if model_path.is_file():
        return model_path

    archive_path = _download_model_archive(model_dir)
    try:
        import zipfile

        with zipfile.ZipFile(archive_path, "r") as zf:
            members = [name for name in zf.namelist() if name.endswith(".pth")]
            if not members:
                raise RuntimeError("RetinaFace archive does not contain a .pth weight file.")
            with zf.open(members[0], "r") as src, model_path.open("wb") as dst:
                dst.write(src.read())
    except Exception as exc:
        raise RuntimeError(
            "Failed to extract RetinaFace PyTorch weights from the downloaded archive."
        ) from exc

    return model_path


def _build_resnet50_backbone() -> nn.Module:
    try:
        return models.resnet50(weights=None)
    except TypeError:
        return models.resnet50(pretrained=False)


def _resize_longest_side(image: np.ndarray, max_size: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if max_size <= 0 or longest <= max_size:
        return image, 1.0

    scale = float(max_size) / float(longest)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    pil_image = Image.fromarray(image, mode="RGB")
    resized = pil_image.resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8), scale


def _normalize_image(image: np.ndarray) -> torch.Tensor:
    arr = image.astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)


def _prior_box(image_size: tuple[int, int], device: torch.device) -> torch.Tensor:
    feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in STEPS]
    anchors: list[float] = []
    for k, feature_map in enumerate(feature_maps):
        min_sizes = MIN_SIZES[k]
        for i in range(feature_map[0]):
            for j in range(feature_map[1]):
                for min_size in min_sizes:
                    s_kx = min_size / image_size[1]
                    s_ky = min_size / image_size[0]
                    cx = (j + 0.5) * STEPS[k] / image_size[1]
                    cy = (i + 0.5) * STEPS[k] / image_size[0]
                    anchors.extend([cx, cy, s_kx, s_ky])
    return torch.tensor(anchors, dtype=torch.float32, device=device).view(-1, 4)


def _decode_boxes(loc: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * VARIANCE[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * VARIANCE[1]),
        ),
        dim=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def _decode_landmarks(pre: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            priors[:, :2] + pre[:, :2] * VARIANCE[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * VARIANCE[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * VARIANCE[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * VARIANCE[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * VARIANCE[0] * priors[:, 2:],
        ),
        dim=1,
    )


class _ClassHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 2) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class _BboxHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 2) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class _LandmarkHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 2) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 10, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


def _conv_bn(inp: int, oup: int, stride: int = 1, leaky: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def _conv_bn_no_relu(inp: int, oup: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def _conv_bn1x1(inp: int, oup: int, stride: int, leaky: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class _SSH(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        if out_channel % 4 != 0:
            raise ValueError(f"expected out_channel % 4 == 0, got {out_channel}")
        leaky = 0.1 if out_channel <= 64 else 0.0
        self.conv3X3 = _conv_bn_no_relu(in_channel, out_channel // 2, stride=1)
        self.conv5X5_1 = _conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = _conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)
        self.conv7X7_2 = _conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = _conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv3x3 = self.conv3X3(x)
        conv5x5_1 = self.conv5X5_1(x)
        conv5x5 = self.conv5X5_2(conv5x5_1)
        conv7x7_2 = self.conv7X7_2(conv5x5_1)
        conv7x7 = self.conv7x7_3(conv7x7_2)
        return F.relu(torch.cat([conv3x3, conv5x5, conv7x7], dim=1))


class _FPN(nn.Module):
    def __init__(self, in_channels_list: list[int], out_channels: int) -> None:
        super().__init__()
        leaky = 0.1 if out_channels <= 64 else 0.0
        self.output1 = _conv_bn1x1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = _conv_bn1x1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = _conv_bn1x1(in_channels_list[2], out_channels, stride=1, leaky=leaky)
        self.merge1 = _conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = _conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, x: Dict[str, torch.Tensor]) -> list[torch.Tensor]:
        features = list(x.values())
        output1 = self.output1(features[0])
        output2 = self.output2(features[1])
        output3 = self.output3(features[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = self.merge2(output2 + up3)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = self.merge1(output1 + up2)
        return [output1, output2, output3]


class _RetinaFaceNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = _build_resnet50_backbone()
        self.body = _utils.IntermediateLayerGetter(
            backbone,
            {"layer2": 1, "layer3": 2, "layer4": 3},
        )
        self.fpn = _FPN([512, 1024, 2048], 256)
        self.ssh1 = _SSH(256, 256)
        self.ssh2 = _SSH(256, 256)
        self.ssh3 = _SSH(256, 256)
        self.ClassHead = nn.ModuleList([_ClassHead(256, 2) for _ in range(3)])
        self.BboxHead = nn.ModuleList([_BboxHead(256, 2) for _ in range(3)])
        self.LandmarkHead = nn.ModuleList([_LandmarkHead(256, 2) for _ in range(3)])

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.body(inputs)
        fpn = self.fpn(out)
        features = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        landmark_regressions = torch.cat(
            [self.LandmarkHead[i](feature) for i, feature in enumerate(features)],
            dim=1,
        )
        return bbox_regressions, classifications, landmark_regressions


@dataclass(frozen=True)
class TorchRetinaFaceAnnotation:
    bbox: list[float]
    score: float
    landmarks: list[list[float]]


class TorchRetinaFace:
    def __init__(
        self,
        *,
        device: str = "cpu",
        max_size: int = DEFAULT_MAX_SIZE,
        model_dir: str | Path | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.max_size = int(max_size)
        self.model_dir = default_model_dir() if model_dir is None else Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.model = _RetinaFaceNetwork().to(self.device)
        self._load_weights()
        self.model.eval()

    def _load_weights(self) -> None:
        try:
            model_path = _ensure_model_file(self.model_dir)
            state_dict = torch.load(model_path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(
                "Failed to download or load RetinaFace PyTorch weights. "
                f"Expected model={MODEL_NAME}. Ensure network access or pre-populate {self.model_dir}."
            ) from exc
        self.model.load_state_dict(state_dict)

    def predict(
        self,
        image: np.ndarray,
        *,
        confidence_threshold: float,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    ) -> list[TorchRetinaFaceAnnotation]:
        original_height, original_width = image.shape[:2]
        resized_image, _scale = _resize_longest_side(image, self.max_size)
        resized_height, resized_width = resized_image.shape[:2]

        image_tensor = _normalize_image(resized_image).to(self.device)
        priors = _prior_box((resized_height, resized_width), self.device)
        scale_bboxes = torch.tensor(
            [resized_width, resized_height, resized_width, resized_height],
            dtype=torch.float32,
            device=self.device,
        )
        scale_landmarks = torch.tensor(
            [resized_width, resized_height] * 5,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            loc, conf, land = self.model(image_tensor)
            conf = F.softmax(conf, dim=-1)

        boxes = _decode_boxes(loc[0], priors) * scale_bboxes
        scores = conf[0][:, 1]
        landmarks = _decode_landmarks(land[0], priors) * scale_landmarks

        valid_index = torch.where(scores > confidence_threshold)[0]
        if valid_index.numel() == 0:
            return []

        boxes = boxes[valid_index]
        landmarks = landmarks[valid_index]
        scores = scores[valid_index]

        keep = nms(boxes, scores, nms_threshold)
        boxes = boxes[keep].cpu().numpy()
        scores = scores[keep].cpu().numpy().astype(float)
        landmarks = landmarks[keep].cpu().numpy().reshape(-1, 5, 2)

        scale_x = float(original_width) / float(resized_width)
        scale_y = float(original_height) / float(resized_height)
        boxes[:, 0] *= scale_x
        boxes[:, 2] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 3] *= scale_y
        landmarks[:, :, 0] *= scale_x
        landmarks[:, :, 1] *= scale_y

        annotations: list[TorchRetinaFaceAnnotation] = []
        for box, score, points in zip(boxes, scores, landmarks):
            x_min, y_min, x_max, y_max = box.astype(np.float32)
            x_min = float(np.clip(x_min, 0.0, max(0.0, original_width - 1.0)))
            y_min = float(np.clip(y_min, 0.0, max(0.0, original_height - 1.0)))
            x_max = float(np.clip(x_max, x_min + 1.0, max(x_min + 1.0, original_width - 1.0)))
            y_max = float(np.clip(y_max, y_min + 1.0, max(y_min + 1.0, original_height - 1.0)))
            if x_max <= x_min or y_max <= y_min:
                continue

            ordered_points = points.astype(np.float32).copy()
            if ordered_points.shape != (5, 2):
                continue

            if ordered_points[0, 0] > ordered_points[1, 0]:
                ordered_points[[0, 1]] = ordered_points[[1, 0]]
            if ordered_points[3, 0] > ordered_points[4, 0]:
                ordered_points[[3, 4]] = ordered_points[[4, 3]]

            annotations.append(
                TorchRetinaFaceAnnotation(
                    bbox=[x_min, y_min, x_max, y_max],
                    score=float(score),
                    landmarks=ordered_points.tolist(),
                )
            )

        return annotations
