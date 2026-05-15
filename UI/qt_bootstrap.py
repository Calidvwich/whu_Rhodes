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


def prepare_qt_runtime() -> None:
    """
    Force Qt to prefer the PyQt5 plugin tree instead of OpenCV's vendored Qt
    plugins, and prefer Wayland under WSLg where xcb dependencies are commonly
    incomplete.
    """
    if not sys.platform.startswith("linux"):
        return

    plugins_dir = _find_pyqt_plugins_dir()
    if plugins_dir is not None:
        plugin_path = str(plugins_dir)
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", plugin_path)
        os.environ.setdefault("QT_PLUGIN_PATH", plugin_path)

    if _running_in_wslg():
        os.environ.setdefault("QT_QPA_PLATFORM", "wayland")
