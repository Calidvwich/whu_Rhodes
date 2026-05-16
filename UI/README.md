# UI 模块运行说明

## 重要说明（请先读）

- 之前版本曾**误上传 venv**，若你本地还是旧 clone，请 **`git pull` 到最新**，并确认仓库里**没有**把虚拟环境目录提交上来。
- 本地虚拟环境目录名建议使用 **`.venv`**（与仓库根目录 `.gitignore` 中 `./UI/.venv/` 一致，避免被误提交）。

## Python 版本

- 推荐 **Python 3.12**（例如 **3.12.9**）。
- **不要使用 3.14 等实验版本**，否则 PyQt5 / sip 等可能出现**编译或安装失败**。
- 检查：`python --version` 应显示 `3.12.x`。若不是，请安装 3.12 后重新建虚拟环境。

## 为什么 `.\.venv\Scripts\python.exe` 找不到？

仓库已忽略 **`UI/.venv/`**，克隆后**默认没有**虚拟环境，需要在本机**创建一次**（见下文）。

## 首次运行（PowerShell）

在 **`whu_Rhodes/UI`** 目录下执行（将路径换成你本机实际路径即可）：

```powershell
cd d:\mowang\2026_1\ZongHeXiangMu\whu_Rhodes\UI

# 使用 Python 3.12 创建虚拟环境（推荐）
py -3.12 -m venv .venv
# 若未安装 py 启动器，可改为指向 3.12 的 python 全路径：
# "C:\Path\To\Python312\python.exe" -m venv .venv

.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 人脸检测/对齐与 FaceNet 特征依赖 torch 等（与仓库 face_engine 一致）
pip install -r ../face_engine/requirements.txt

python face_ui_pyqt5.py
```

摄像头说明：

- 在 Windows 原生环境运行时，程序会依次尝试 `DirectShow`、`Media Foundation` 和 OpenCV 自动后端。
- 如果你通过 VS Code Remote WSL 在 Linux 环境里运行，只有当 WSL 内能看到 `/dev/video*` 时摄像头才可用。

Linux / WSL 补充说明：

```bash
env QT_QPA_PLATFORM=wayland python face_ui_pyqt5.py
```

若仍出现 `Could not load the Qt platform plugin "xcb"`，请安装缺失系统库：

```bash
sudo apt update
sudo apt install -y libxcb-icccm4 libxcb-keysyms1
```

如果程序提示 WSL2 中没有任何 `/dev/video*` 设备，请直接切换到 Windows 原生 Python/Conda 环境运行；这类情况不是 UI 代码本身能修复的。

若执行策略禁止 `Activate.ps1`，可不用激活，直接用完整路径：

```powershell
cd d:\mowang\2026_1\ZongHeXiangMu\whu_Rhodes\UI
py -3.12 -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\pip.exe install -r ..\face_engine\requirements.txt
.\.venv\Scripts\python.exe face_ui_pyqt5.py
```

## 入口与数据库

- 主界面（PyQt5）：**`face_ui_pyqt5.py`**
- 数据库路径在 `face_ui_pyqt5.py` 中指向仓库内 **`../database/`**（与 `UI` 同级）。

## 摄像头来源选择

PyQt5 主界面支持两类摄像头来源：

- 本地摄像头：点击“刷新设备”会在后台枚举 `camera:0` 到 `camera:8`，避免反复同步打开 DirectShow 设备导致界面崩溃。
- 手机浏览器摄像头：点击“启用手机摄像头”后，电脑端会启动本地 HTTPS 服务并在弹窗中显示二维码。手机和电脑需要连接同一 Wi-Fi 或电脑热点，手机扫码后首次访问可能需要接受自签名证书提示，然后允许浏览器访问摄像头。
- 检测到手机开始传输画面后，电脑端扫码弹窗会自动关闭；再次点击“停止手机摄像头”会停止手机采集并恢复按钮状态。
- 手机网页默认使用前置摄像头，可在手机网页中切换前置/后置镜头，也可以切换竖屏/横屏发送方向；是否支持镜头切换取决于手机浏览器的 `getUserMedia` 实现。
- 手机画面会以 `mobile-browser` 作为设备信息写入识别流程；人脸识别仍在电脑端完成。
- 启动手机扫码功能需要额外依赖 `cryptography` 和 `qrcode[pil]`，已写入 `UI/requirements.txt`。


