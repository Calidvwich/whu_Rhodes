# Database 模块说明

本模块负责用户信息、512 维人脸特征向量、识别阈值和识别日志的本地持久化。当前联调版本使用 SQLite，默认数据库文件为 `database/data/app.sqlite3`。

## 环境准备

建议直接复用项目联调用的 Conda 环境：

```powershell
cd <你的 whu_Rhodes 仓库目录>
conda activate whu_rhodes_ui
pip install -r database/requirements.txt
```

如果从零创建环境，Python 版本建议使用 3.10 或 3.11。

## 初始化数据库

```powershell
python database/scripts/init_db.py
```

初始化脚本会读取 `database/src/db/schema.sql`，创建 `user`、`face_feature`、`recognition_log`、`system_config` 四张表，并写入默认阈值：

```text
threshold = 0.65
```

`database/data/app.sqlite3` 是本地开发数据库，已加入忽略规则，不建议提交到 Git。其他同学拉取代码后需要在自己的电脑上重新执行初始化脚本。

## 联调入口

推荐 UI 或其他模块通过 `src.service.FaceRecService` 调用数据库能力：

```python
from pathlib import Path
import sys

sys.path.insert(0, Path("database").resolve().as_posix())

from src.service import FaceRecService

service = FaceRecService(Path("database/data/app.sqlite3"))
service.init_db()

enrolled = service.enroll("alice", feature_vector, image_path="alice.jpg")
result = service.recognize(feature_vector, device_info="desktop-camera")
```

当前服务层提供：

- `enroll(...)`：创建或复用用户，并写入 512 维人脸特征。
- `recognize(...)`：读取全部 active 特征，按余弦相似度匹配，并写入识别日志。
- `get_logs_by_user(...)`：按用户查询识别日志。
- `get_logs_by_time_range(...)`：按时间范围查询识别日志。
- `get_logs_by_device_info(...)`：按设备来源查询识别日志，例如 `desktop-camera`、`mobile-browser`。
- `deactivate_feature(...)`：停用单条特征。
- `deactivate_features_by_user(...)`：停用某个用户的全部 active 特征。

## 验证脚本

数据库烟测：

```powershell
python database/scripts/smoke_test.py
```

该脚本使用临时 SQLite 数据库，不会污染 `database/data/app.sqlite3`。它会覆盖初始化、注册、命中识别、陌生人拒绝、日志查询、特征停用几个关键路径。

阈值说明脚本：

```powershell
python database/scripts/threshold_report.py --threshold 0.65
```

该脚本输出一组固定随机种子的余弦相似度样例，用来解释阈值判断逻辑。注意它是合成向量 sanity check，最终报告里如果需要更强证据，应替换为真实人脸特征向量样本。

旧 demo 仍可运行：

```powershell
python database/scripts/demo_compare.py
```

如果算法模块已经导出了 `{"vector":[512 floats]}` 格式的 JSON，也可以继续使用旧的 JSON 联调脚本：

```powershell
python database/scripts/feature_from_json.py enroll --username alice --vector-json database/examples/vector_512.json
python database/scripts/feature_from_json.py recognize --vector-json database/examples/vector_512.json
```

## 数据提交策略

- 代码、schema、脚本、README 可以提交。
- `.venv/`、`__pycache__/`、`.pytest_cache/`、`database/data/`、`*.sqlite3` 不提交。
- 如果本地数据库坏了，可以删除 `database/data/app.sqlite3` 后重新执行 `python database/scripts/init_db.py`。

## 后续迁移 MySQL 的位置

当前 DAO 层集中在 `database/src/db/dao.py`，连接封装在 `database/src/db/connection.py`。后续如果要切换到 MySQL，优先替换连接层和 SQL 方言，尽量保持 `FaceRecService` 的接口不变，这样 UI 和算法模块不需要大改。
