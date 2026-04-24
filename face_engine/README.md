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
- FaceNet / MTCNN 代码来自仓库内的 `facenet_pytorch` git submodule
- 不再单独安装 `facenet-pytorch` pip 包

## 2. 输入输出契约

- 输入：`RGB` 对齐后人脸图，推荐尺寸 `160x160`
- 输出：JSON 文件，格式为 `{"vector":[512个float]}`

## 3. 命令行使用

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

## 4. Python 调用方式

```python
from face_engine import extract_from_aligned_face, save_vector_json

vec = extract_from_aligned_face("aligned_face.jpg", l2_normalize_output=True)
save_vector_json(vec, "aligned_face.vector.json")
```
