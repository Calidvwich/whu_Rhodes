# whu_Rhodes

计算机系统设计项目仓库。当前按职责拆成两个主要模块：

- `face_engine/`：人脸特征提取算法层，输入对齐后的人脸图，输出 `{"vector":[512 floats]}`
- `database/`：数据持久层与特征库联调层，负责向量存储、相似度比对、识别日志

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
  - `python face_engine/scripts/extract_feature.py single ...`
  - `python face_engine/scripts/verify_extractor.py --download-samples`
- 详细说明见 [face_engine/README.md](/home/orangeisland66/桌面/whu_Rhodes/face_engine/README.md)

### `database/`

- 负责 SQLite 建表、DAO、向量入库、余弦相似度比对、识别日志
- 通过 JSON 向量和算法层联调：
  - `python database/scripts/feature_from_json.py enroll ...`
  - `python database/scripts/feature_from_json.py recognize ...`
- 详细说明见 [database/README.md](/home/orangeisland66/桌面/whu_Rhodes/database/README.md)

## 最小联调流程

```bash
# 1) 提取 512 维向量
python face_engine/scripts/extract_feature.py single \
  --image path/to/aligned_face.jpg \
  --out-json database/examples/face_a.json

# 2) 初始化数据库
python database/scripts/init_db.py

# 3) 录入向量
python database/scripts/feature_from_json.py enroll \
  --username alice \
  --vector-json database/examples/face_a.json \
  --image-path path/to/aligned_face.jpg

# 4) 用向量识别
python database/scripts/feature_from_json.py recognize \
  --vector-json database/examples/face_a.json \
  --device-info demo_camera
```
