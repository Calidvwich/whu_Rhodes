# AGENTS.md

本文件给后续代码代理使用。修改本仓库前请先阅读，优先保持现有模块边界和数据契约。

## 项目概览

`whu_Rhodes` 是一个课程设计性质的人脸识别系统，主链路为：

1. `UI/face_ui_pyqt5.py` 采集摄像头帧或读取图片。
2. `face_engine/` 使用仓库内 `facenet_pytorch` 的 MTCNN 做人脸检测与对齐，再用 FaceNet 提取 512 维特征向量。
3. `database/` 将特征向量写入 SQLite，或对特征库做余弦相似度匹配，并写入识别日志。
4. UI 按 `code / success / message / data / timestamp` 响应格式展示结果。

主要目录：

- `UI/`：用户界面与当前业务编排层。主入口是 `UI/face_ui_pyqt5.py`，`UI/face_ui.py` 是较早的 Tkinter 版本。
- `UI/camera_widget.py`：PyQt5 摄像头采集与帧显示组件，依赖 OpenCV。
- `face_engine/`：算法层源码、CLI 包装脚本、示例图片和向量 JSON。
- `database/`：SQLite schema、DAO、向量编码、匹配逻辑和数据库联调脚本。
- `facenet_pytorch/`：项目内的 FaceNet/MTCNN 依赖代码。除非任务明确要求升级算法依赖，否则不要随意改动。

## 环境与依赖

推荐 Python 3.12。现有说明偏向 Conda：

```powershell
conda create -n whu_rhodes_ui python=3.12 -y
conda run -n whu_rhodes_ui python -m pip install -r UI\requirements.txt
conda run -n whu_rhodes_ui python -m pip install -r face_engine\requirements.txt
```

也可以使用本地 venv，但不要提交虚拟环境目录。首次调用 FaceNet 可能联网下载预训练权重，缓存目录为 `face_engine/.model_cache/`。

重要依赖文件：

- `UI/requirements.txt`：PyQt5、OpenCV、NumPy、bcrypt。
- `face_engine/requirements.txt`：torch、torchvision、Pillow、NumPy 等算法依赖。
- `database/requirements.txt`：NumPy。

## 常用命令

从仓库根目录运行：

```powershell
# 初始化 SQLite 数据库
conda run -n whu_rhodes_ui python database\scripts\init_db.py

# 验证算法层能输出 512 维、L2 归一化向量
conda run -n whu_rhodes_ui python -B -c "import sys, numpy as np; from pathlib import Path; sys.path.insert(0, str(Path('face_engine/src').resolve())); from face_engine import extract_from_aligned_face; vec = extract_from_aligned_face(r'face_engine\examples\aligned_face.png', model_cache=r'face_engine\.model_cache'); print(vec.shape, float(np.linalg.norm(vec)))"

# 启动 PyQt5 UI
conda run -n whu_rhodes_ui python UI\face_ui_pyqt5.py
```

算法层 CLI：

```powershell
python face_engine\scripts\preprocess_face.py single --image path\to\raw_face.jpg --out-image face_engine\examples\aligned_face.png
python face_engine\scripts\extract_feature.py single --image face_engine\examples\aligned_face.png --out-json database\examples\face_a.json
python face_engine\scripts\verify_extractor.py --download-samples
```

数据库联调 CLI：

```powershell
python database\scripts\feature_from_json.py enroll --username alice --vector-json database\examples\vector_512.json
python database\scripts\feature_from_json.py recognize --vector-json database\examples\vector_512.json --device-info demo_camera
python database\scripts\inspect_db.py
```

## 核心契约

- 人脸向量固定为 512 维 `float32`。
- 算法层 JSON 输出格式为 `{"vector": [512 floats]}`；数据库脚本也兼容裸数组 `[512 floats]`。
- 对齐人脸默认尺寸为 `160x160` RGB。
- 特征写库前应保持 L2 归一化；现有提取接口默认归一化。
- 数据库中 `face_feature.feature_vector` 是 `512 * float32` 的 BLOB，不要改成字符串 JSON，除非同步迁移 DAO 与脚本。
- 匹配逻辑使用余弦相似度，默认阈值来自 `system_config.threshold`，初始值为 `0.80`。
- UI 业务接口返回统一结构：`code`、`success`、`message`、`data`、`timestamp`。

## 模块约定

### UI

- 主力版本是 `UI/face_ui_pyqt5.py`。除非任务明确要求维护 Tkinter 旧版，不要把新功能只加到 `UI/face_ui.py`。
- `FaceBusinessService` 当前承担登录、注册、用户管理、特征录入、识别、配置更新等业务编排。
- `FaceAlgorithm` 通过 `face_engine/src` 导入算法接口，并使用 `face_engine/.model_cache` 作为模型缓存。
- 摄像头帧来自 OpenCV，通常是 BGR；送入算法层前需要转 RGB。
- 本地演示账号由 UI 业务层自动初始化：`admin/admin123`、`user1/123456`。

### face_engine

- 公共 Python 接口由 `face_engine/src/face_engine/__init__.py` 暴露：
  `preprocess_face_image`、`preprocess_face`、`save_preprocessed_face`、`extract_from_aligned_face`、`save_vector_json`。
- `face_preprocessor.py` 负责 MTCNN 检测、选择最大人脸、5 点相似变换对齐。
- `facenet_extractor.py` 负责 FaceNet 推理、尺寸兜底调整、L2 归一化和 JSON 保存。
- `scripts/*.py` 只是给 `src` 中 CLI 模块加路径的薄包装，业务逻辑应放在 `src/face_engine/*_cli.py` 或核心模块里。

### database

- schema 在 `database/src/db/schema.sql`。
- 连接逻辑在 `database/src/db/connection.py`；DAO 在 `database/src/db/dao.py`。
- 向量编码和校验在 `database/src/feature/vector_codec.py`。
- 匹配逻辑在 `database/src/feature/matcher.py`。
- 本地数据库路径默认为 `database/data/app.sqlite3`，这是开发数据文件，不要依赖它在干净 clone 中存在。

## 开发注意事项

- 不要提交 `UI/.venv/`、`database/.venv/`、`__pycache__/`、`*.pyc`、`face_engine/.model_cache/`、本地 SQLite 数据库等生成物。
- 当前工作树可能已有用户产生的文件或缓存；不要清理、回滚或重置用户未要求处理的改动。
- 修改跨模块契约时，要同步更新 README、CLI、UI 业务层和数据库脚本。
- 尽量保持路径从仓库根目录可运行；脚本中已有 `sys.path.insert` 的地方不要改成依赖全局安装包。
- bcrypt 是可选导入；缺失时 UI 会退回 `sha256$` 哈希格式。改登录逻辑时要兼容已有数据。
- OpenCV、NumPy、torch 都是可选/重型依赖场景中容易失败的点。UI 层应继续返回清晰的错误响应，而不是让异常穿透到界面事件循环。
- 如果新增测试或验证脚本，优先使用仓库自带示例图片和向量，避免要求真实摄像头。

## 建议验证

按改动范围选择验证：

- 只改数据库层：运行 `python database\scripts\init_db.py`，再运行相关 `feature_from_json.py` 或 `demo_compare.py`。
- 只改算法层：用 `face_engine\examples\aligned_face.png` 验证输出形状和范数，必要时运行 `verify_extractor.py`。
- 改 UI 业务层：至少启动 `python UI\face_ui_pyqt5.py`，验证登录、注册、管理员用户管理入口；涉及识别时再验证图片/摄像头流程。
- 改 schema 或响应结构：同时做一次 `UI -> face_engine -> database -> UI` 的最小链路验证。

