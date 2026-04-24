# 基于深度学习的人脸识别系统（最小后端骨架）

本目录当前提供一个最小可运行的数据库与特征库联调实现，输入应为算法层产出的 `512` 维向量 JSON：

- SQLite 数据库（免安装，先跑通流程）
- 建表脚本（`USER / FACE_FEATURE / RECOGNITION_LOG / SYSTEM_CONFIG`）
- 数据访问层（DAO）
- 512 维特征向量（`float32`）的序列化/反序列化存取
- 余弦相似度比对 + 阈值判定

## 1. 环境准备

建议 Python `3.10+`。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 输入输出契约

- 输入：JSON 向量文件，格式为 `{"vector":[512个float]}`，也兼容直接传 `[512个float]`
- 输出：
  - `enroll`：向数据库写入用户和特征
  - `recognize`：返回 `matched_user_id / max_similarity / matched / log_id`

本目录不负责做人脸检测、裁剪、对齐或特征提取；这些工作应由上游算法模块完成。

## 3. 初始化数据库并跑一个最小闭环

```bash
python scripts/init_db.py
python scripts/demo_compare.py
```

你会看到：
- 数据库初始化成功
- 插入 1 个用户、N 条随机特征
- 用一条“接近某用户”的向量做比对，输出 `matched_user_id / similarity / matched`

## 4. 使用 JSON 向量联调

```bash
# 录入：把某个用户的 512 维向量写入 face_feature
python scripts/feature_from_json.py enroll --username alice --vector-json examples/vector_512.json

# 识别：读取 512 维向量，全库比对，写 recognition_log
python scripts/feature_from_json.py recognize --vector-json examples/vector_512.json
```

如果需要从图片提取向量，请使用仓库根目录下的 `face_engine/` 模块，生成 `{"vector":[512 floats]}` 后再交给本目录联调。
