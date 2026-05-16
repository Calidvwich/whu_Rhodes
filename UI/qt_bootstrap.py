import importlib.util
import os
import sys
from pathlib import Path


def _find_pyqt_plugins_dir() -> Path | None:
    spec = importlib.util.find_spec("PyQt5")
    if spec is None or spec.origin is None:
        return None
    plugins_dir = Path(spec.origin).resolve().parent / "Qt5" / "plugins"
    if plugins_dir.is_dir():
        return plugins_dir
    return None


def _running_in_wslg() -> bool:
    return bool(os.environ.get("WSL_DISTRO_NAME")) and bool(os.environ.get("WAYLAND_DISPLAY"))


def _running_in_wayland_session() -> bool:
    return (
        os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
        or bool(os.environ.get("WAYLAND_DISPLAY"))
    )


def prepare_qt_runtime() -> None:
    """
    Force Qt to prefer the PyQt5 plugin tree instead of OpenCV's vendored Qt
    plugins, and prefer Wayland when the desktop session is already Wayland.
    """
    if not sys.platform.startswith("linux"):
        return

    plugins_dir = _find_pyqt_plugins_dir()
    if plugins_dir is not None:
        platforms_dir = plugins_dir / "platforms"
        if platforms_dir.is_dir():
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms_dir)
        os.environ["QT_PLUGIN_PATH"] = str(plugins_dir)

    if _running_in_wslg() or _running_in_wayland_session():
        os.environ.setdefault("QT_QPA_PLATFORM", "wayland")
