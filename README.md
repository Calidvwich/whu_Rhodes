# whu_Rhodes

计算机系统设计项目仓库。当前按职责拆成三个主要模块：

- `face_engine/`：人脸特征提取算法层，输入对齐后的人脸图，输出 `{"vector":[512 floats]}`
- `database/`：数据持久层与特征库联调层，负责向量存储、相似度比对、识别日志
- `UI/`：用户交互层，主要负责将不同模块之间的数据在不同端之间传输，同时实现直观图形化的用户交互功能

## 克隆方式

请使用 submodule 方式克隆：

```bash
git clone --recurse-submodules git@github.com:shanren7/real_time_face_recognition.git
```

或：

```bash
git clone --recurse-submodules https://github.com/shanren7/real_time_face_recognition.git
```

如果已经克隆但未初始化子模块：

```bash
git submodule update --init --recursive
```

如果需要更新子模块到最新远端版本：

```bash
git submodule update --remote
```

## 目录说明

### `face_engine/`

- 使用仓库内 `facenet_pytorch` submodule 提供的 FaceNet / MTCNN
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
