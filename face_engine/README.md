# Face Engine

本目录负责“对齐人脸图 -> 512 维特征向量 JSON”的算法层实现。

## 1. 环境准备

请从仓库根目录以 submodule 方式克隆：

```bash
git clone --recurse-submodules git@github.com:shanren7/real_time_face_recognition.git
```

安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r face_engine/requirements.txt
```

说明：
- FaceNet 代码来自仓库内的 `facenet_pytorch` git submodule
- 人脸检测与关键点定位改用仓库内置的纯 `PyTorch RetinaFace`
- 首次使用 RetinaFace 可能联网下载模型权重；若离线环境缺少权重，将无法完成检测

## 2. 预处理模块

仓库现在提供原始图片预处理 CLI，用于给 `face_engine` 生成稳定输入：

单图预处理：

```bash
python face_engine/scripts/preprocess_face.py single \
  --image path/to/raw_photo.jpg \
  --out-image face_engine/examples/aligned_face.png
```

批量预处理：

```bash
python face_engine/scripts/preprocess_face.py batch \
  --input-dir path/to/raw_images \
  --out-dir face_engine/examples/aligned_faces
```

说明：
- 使用纯 `PyTorch RetinaFace` 做人脸检测和 5 点关键点定位
- 检测到多张人脸时，默认选择面积最大的那一张
- 基于 5 点关键点做相似变换对齐，不是仅按框裁剪
- 输出固定为 `RGB` 对齐人脸图，默认尺寸 `160x160`
- 如果检测失败，脚本会直接报错并给出失败图片路径

## 3. 输入输出契约

- 输入：`RGB` 对齐后人脸图，推荐尺寸 `160x160`
- 输出：JSON 文件，格式为 `{"vector":[512个float]}`

## 4. 命令行使用

单图提取：

```bash
python face_engine/scripts/extract_feature.py single \
  --image path/to/aligned_face.jpg \
  --out-json face_engine/examples/aligned_face.vector.json
```

批量提取：

```bash
python face_engine/scripts/extract_feature.py batch \
  --input-dir path/to/aligned_faces \
  --out-dir face_engine/examples/batch_vectors
```

清单提取：

```bash
python face_engine/scripts/extract_feature.py batch \
  --manifest face_engine/examples/face_samples_manifest.example.json \
  --out-dir face_engine/examples/batch_vectors
```

公开样例验证：

```bash
python face_engine/scripts/verify_extractor.py --download-samples
```

本地清单验证：

```bash
python face_engine/scripts/verify_extractor.py \
  --manifest face_engine/examples/face_samples_manifest.example.json \
  --out-dir face_engine/examples/verify_vectors_local
```

如果首次运行预处理时报错，请先确认：

- 已安装 `face_engine/requirements.txt` 中的 `torch`、`torchvision` 等依赖
- 当前环境允许 RetinaFace 首次下载模型权重
- 如离线运行，需要预先准备 RetinaFace 所需权重缓存

## 5. Python 调用方式

```python
from face_engine import (
    extract_from_aligned_face,
    preprocess_face_image,
    save_vector_json,
)

aligned = preprocess_face_image("raw_photo.jpg")
aligned.save("aligned_face.png")
vec = extract_from_aligned_face("aligned_face.png", l2_normalize_output=True)
save_vector_json(vec, "aligned_face.vector.json")
```
