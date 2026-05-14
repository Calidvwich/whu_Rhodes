# UI 模块运行说明

## 为什么 `.\.venv\Scripts\python.exe` 找不到？

仓库根目录的 `.gitignore` 已忽略 **`UI/.venv/`**，克隆下来的项目里**默认没有**虚拟环境，需要你在本机创建一次。

## 首次运行（PowerShell）

在 **`whu_Rhodes/UI`** 目录下执行：

```powershell
cd d:\mowang\2026_1\ZongHeXiangMu\whu_Rhodes\UI

# 任选其一：用当前 python 创建 venv（建议 3.10+）
py -3 -m venv .venv
# 或
python -m venv .venv

.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 人脸检测/对齐与 FaceNet 特征依赖 torch 等（与仓库 face_engine 一致）
pip install -r ../face_engine/requirements.txt

python face_ui_pyqt5.py
```

若执行策略禁止激活脚本，可不用 activate，直接用完整路径：

```powershell
cd d:\mowang\2026_1\ZongHeXiangMu\whu_Rhodes\UI
py -3 -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\python.exe face_ui_pyqt5.py
```

## 入口

- 主界面（PyQt5）：`face_ui_pyqt5.py`

数据库路径在 `face_ui_pyqt5.py` 中指向仓库内 **`../database/`**（与 `UI` 同级），无需再改 Cursor 工作区根目录下的 `src`。
