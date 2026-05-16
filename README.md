# whu_Rhodes

计算机系统设计项目仓库。当前按职责拆成三个主要模块：

- `face_engine/`：人脸特征提取算法层，输入对齐后的人脸图，输出 `{"vector":[512 floats]}`
- `database/`：数据持久层与特征库联调层，负责向量存储、相似度比对、识别日志
- `UI/`：用户交互层，主要负责将不同模块之间的数据在不同端之间传输，同时实现直观图形化的用户交互功能

## 克隆方式

请使用 submodule 方式克隆：

```bash
git clone --recurse-submodules https://github.com/Calidvwich/whu_Rhodes.git
```

如果已经克隆但未初始化子模块：

```bash
git submodule update --init --recursive
```

如果本地已有旧版本，请先更新主仓与子模块：

```bash
git pull
git submodule update --init --recursive
```

如果需要更新子模块到最新远端版本：

```bash
git submodule update --remote
```

## 目录说明

### `face_engine/`

- 使用仓库内 `facenet_pytorch` submodule 提供的 FaceNet
- 人脸检测与关键点定位使用纯 PyTorch RetinaFace
- 提供 Python 接口与 CLI：
  - `python face_engine/scripts/preprocess_face.py single ...`
  - `python face_engine/scripts/extract_feature.py single ...`
  - `python face_engine/scripts/verify_extractor.py --download-samples`
- 详细说明见 [face_engine/README.md](face_engine/README.md)

### `database/`

- 负责 SQLite 建表、DAO、向量入库、余弦相似度比对、识别日志
- 通过 JSON 向量和算法层联调：
  - `python database/scripts/feature_from_json.py enroll ...`
  - `python database/scripts/feature_from_json.py recognize ...`
- 详细说明见 [database/README.md](database/README.md)

## 四模块联调运行说明

当前主链路已经按 `UI -> face_engine -> database -> UI` 跑通：

1. `UI/face_ui_pyqt5.py` 采集或读取图像。
2. `face_engine/` 使用 RetinaFace 做人脸检测与 5 点对齐，并用 FaceNet 提取 `512` 维特征向量。
3. `database/` 将特征向量写入 SQLite，或与特征库做余弦相似度比对并写识别日志。
4. UI 根据业务接口返回的 `code / success / message / data / timestamp` 展示结果。

推荐使用 Conda 新建本地环境，不要使用别人机器拷贝来的 `.venv`：

```powershell
cd <你的 whu_Rhodes 仓库目录>

conda create -n whu_rhodes_ui python=3.12 -y
conda run -n whu_rhodes_ui python -m pip install -r UI/requirements.txt
conda run -n whu_rhodes_ui python -m pip install -r face_engine/requirements.txt
```

初始化本地数据库：

```powershell
conda run -n whu_rhodes_ui python database/scripts/init_db.py
```

首次运行算法层时会自动下载 FaceNet 预训练权重；PyTorch RetinaFace 也可能在首次检测时下载模型权重。`face_engine/.model_cache/` 用于当前仓库内的模型缓存。离线环境如果没有现成权重，预处理会直接报错而不是退化运行。可先用示例图验证特征提取：

```powershell
conda run -n whu_rhodes_ui python -B -c "import sys, numpy as np; from pathlib import Path; sys.path.insert(0, str((Path('face_engine') / 'src').resolve())); from face_engine import extract_from_aligned_face; vec = extract_from_aligned_face(Path('face_engine') / 'examples' / 'aligned_face.png', model_cache=Path('face_engine') / '.model_cache'); print(vec.shape, float(np.linalg.norm(vec)))"
```

成功时应输出类似：

```text
(512,) 1.0
```

启动 PyQt5 UI：

```powershell
conda run -n whu_rhodes_ui python UI/face_ui_pyqt5.py
```

关于摄像头：

- 若需要稳定使用本机 Windows 摄像头，优先在 Windows 原生终端中运行上面的命令。
- 当前 UI 在 Windows 下会依次尝试 `DirectShow`、`Media Foundation` 和 OpenCV 自动后端。
- 若你是在 WSL 中运行，只有当 Linux 侧真的出现 `/dev/video*` 设备时，OpenCV 才能打开摄像头。

如果你是在 Linux / WSL 下运行 PyQt5 UI，请额外注意：

- WSLg 环境建议优先走 Wayland：

```bash
env QT_QPA_PLATFORM=wayland conda run -n whu_rhodes_ui python UI/face_ui_pyqt5.py
```

- 若仍报 `Could not load the Qt platform plugin "xcb"`，先补齐系统库：

```bash
sudo apt update
sudo apt install -y libxcb-icccm4 libxcb-keysyms1
```

- 若 UI 明确提示 “WSL2 中没有任何 `/dev/video*` 设备”，说明当前不是代码问题，而是 WSL 环境还没有拿到摄像头设备。此时请优先改为 Windows 原生运行；若必须在 WSL 中使用，请先用 `usbipd-win` 将 USB 摄像头附加到 WSL。
- 当前 UI 已在代码里优先使用 `PyQt5` 自身的 Qt 插件目录，避免 `opencv-python` 自带的 `cv2/qt/plugins` 干扰 PyQt5 启动。

本地演示账号由 UI 业务层自动初始化：

- 管理员：`admin` / `admin123`
- 普通用户：`user1` / `123456`

注意事项：

- `UI/.venv/`、`database/.venv/`、`__pycache__/`、`*.pyc`、`face_engine/.model_cache/` 都不要提交。
- `database/data/app.sqlite3` 是本地开发数据库。换电脑后如果没有数据库文件，先运行 `database/scripts/init_db.py`，再重新录入人脸特征。
- 第一次提取特征需要联网下载模型权重；下载完成后会走本地缓存。
- 当前开发期数据库使用 SQLite，后续如迁移 MySQL，优先改 `database/src/db/connection.py` 与建表 SQL，算法向量接口尽量保持不变。

## 最小联调流程

```bash
# 1) 预处理原始人脸图
python face_engine/scripts/preprocess_face.py single \
  --image path/to/raw_face.jpg \
  --out-image face_engine/examples/aligned_face.png

# 2) 提取 512 维向量
python face_engine/scripts/extract_feature.py single \
  --image face_engine/examples/aligned_face.png \
  --out-json database/examples/face_a.json

# 3) 初始化数据库
python database/scripts/init_db.py

# 4) 录入向量
python database/scripts/feature_from_json.py enroll \
  --username alice \
  --vector-json database/examples/face_a.json \
  --image-path face_engine/examples/aligned_face.png

# 5) 用向量识别
python database/scripts/feature_from_json.py recognize \
  --vector-json database/examples/face_a.json \
  --device-info demo_camera
```
