# 基于深度学习的人脸识别系统（最小后端骨架）

本仓库当前提供一个**从 0 开始可跑通**的最小后端骨架，用于落地第二阶段的“数据持久层 + 特征库比对”：

- SQLite 数据库（免安装，先跑通流程）
- 建表脚本（USER / FACE_FEATURE / RECOGNITION_LOG / SYSTEM_CONFIG）
- 数据访问层（DAO）
- 512 维特征向量（float32）的序列化/反序列化存取
- 余弦相似度比对 + 阈值判定

## 1. 环境准备

建议 Python 3.10+。

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. 初始化数据库并跑一个最小闭环

```bash
python scripts/init_db.py
python scripts/demo_compare.py
```

你会看到：
- 数据库初始化成功
- 插入 1 个用户、N 条随机特征
- 用一条“接近某用户”的向量做比对，输出 `matched_user_id / similarity / matched`

## 3. 下一步怎么扩展

- 使用 JSON 形式的算法输出向量联调（推荐先这样对接）：

```bash
# 录入：把某个用户的 512 维向量写入 face_feature
python scripts/feature_from_json.py enroll --username alice --vector-json examples/vector_512.json

# 识别：读取 512 维向量，全库比对，写 recognition_log
python scripts/feature_from_json.py recognize --vector-json examples/vector_512.json
```

- 把 `scripts/demo_compare.py` 替换为：从算法同学输出的 512 维向量入库、从摄像头帧提取向量做识别（当你们联调方式确定后）
- 把 SQLite 切换为 MySQL：保留接口不变，只替换 `src/db/connection.py` 的连接实现与建表脚本

