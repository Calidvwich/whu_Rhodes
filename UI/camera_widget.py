import importlib
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from qt_bootstrap import prepare_qt_runtime

prepare_qt_runtime()

from PyQt5.QtCore import QRect, QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QLabel, QMessageBox, QVBoxLayout, QWidget

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")


def _optional_import(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


cv2 = _optional_import("cv2")


def _configure_opencv_logging():
    if cv2 is None:
        return
    try:
        cv2.setLogLevel(0)
    except Exception:
        pass


_configure_opencv_logging()


class _ClosedCapture:
    def isOpened(self):
        return False

    def release(self):
        return None


def _is_wsl() -> bool:
    return sys.platform.startswith("linux") and (
        "microsoft" in os.uname().release.lower() or bool(os.environ.get("WSL_DISTRO_NAME"))
    )


def _list_linux_video_devices():
    return sorted(str(path) for path in Path("/dev").glob("video*"))


def _candidate_backends():
    if cv2 is None:
        return [("unavailable", None)]

    candidates = []
    if sys.platform.startswith("win"):
        if hasattr(cv2, "CAP_DSHOW"):
            candidates.append(("DirectShow", cv2.CAP_DSHOW))
        if hasattr(cv2, "CAP_MSMF"):
            candidates.append(("Media Foundation", cv2.CAP_MSMF))
    elif sys.platform.startswith("linux"):
        if hasattr(cv2, "CAP_V4L2"):
            candidates.append(("V4L2", cv2.CAP_V4L2))

    candidates.append(("Auto", None))
    return candidates


def _open_video_capture(camera_index):
    if cv2 is None:
        return None, None, ["OpenCV 未安装"]

    if sys.platform.startswith("linux") and not _list_linux_video_devices():
        return _ClosedCapture(), None, ["Linux 未发现 /dev/video* 设备"]

    errors = []
    for backend_name, backend in _candidate_backends():
        if backend is None:
            cap = cv2.VideoCapture(camera_index)
        else:
            cap = cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            return cap, backend_name, errors
        errors.append(f"{backend_name} 打开失败")
        cap.release()

    return cap, None, errors


def scan_available_cameras(max_index=8, skip_indices=None):
    if cv2 is None:
        return []

    skip = {int(i) for i in (skip_indices or [])}
    if sys.platform.startswith("linux") and not _list_linux_video_devices():
        return []

    devices = []
    for camera_index in range(int(max_index)):
        if camera_index in skip:
            continue

        cap, backend_name, _errors = _open_video_capture(camera_index)
        try:
            if cap is not None and cap.isOpened():
                backend = f" ({backend_name})" if backend_name else ""
                devices.append({
                    "index": camera_index,
                    "name": f"camera:{camera_index}{backend}",
                })
        finally:
            if cap is not None:
                cap.release()

    return devices


def _build_camera_unavailable_message(camera_index, errors):
    prefix = f"camera:{camera_index} 不可用。"
    if _is_wsl():
        devices = _list_linux_video_devices()
        if not devices:
            return (
                f"{prefix} 当前运行在 WSL2，但 Linux 侧没有任何 /dev/video* 设备。"
                "Windows 主机摄像头不会自动暴露给 WSL。"
                "如需直接使用摄像头，请优先在 Windows 原生 Python/Conda 环境运行该 UI；"
                "若必须在 WSL 中使用，请先用 usbipd-win 将 USB 摄像头附加到 WSL。"
            )
        return (
            f"{prefix} WSL 中已看到设备 {', '.join(devices)}，但 OpenCV 打开失败。"
            f"已尝试后端: {', '.join(errors) or '无'}。"
        )

    if sys.platform.startswith("linux"):
        devices = _list_linux_video_devices()
        if not devices:
            return f"{prefix} Linux 下未发现 /dev/video* 设备。"
        return f"{prefix} 已发现设备 {', '.join(devices)}，但 OpenCV 打开失败。已尝试后端: {', '.join(errors) or '无'}。"

    if sys.platform.startswith("win"):
        return (
            f"{prefix} 已尝试后端: {', '.join(errors) or '无'}。"
            "请检查 Windows 相机权限、设备是否被其他应用占用，或切换到原生 Windows 终端运行。"
        )

    return f"{prefix} 已尝试后端: {', '.join(errors) or '无'}。"


def _build_camera_unavailable_detail(camera_index, errors):
    if _is_wsl():
        devices = _list_linux_video_devices()
        if not devices:
            return f"设备: camera:{camera_index}  WSL /dev/video*: 无"
        return f"设备: {', '.join(devices)}  后端: {', '.join(errors) or '-'}"

    if sys.platform.startswith("linux"):
        devices = _list_linux_video_devices()
        if not devices:
            return f"设备: camera:{camera_index}  /dev/video*: 无"
        return f"设备: {', '.join(devices)}  后端: {', '.join(errors) or '-'}"

    return f"设备: camera:{camera_index}  后端: {', '.join(errors) or '-'}"


class CameraThread(QThread):
    frame_ready = pyqtSignal(object)   # numpy frame (BGR)
    camera_lost = pyqtSignal(str, str)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.cap = None
        self.backend_name = None

    def run(self):
        self.running = True
        if cv2 is None:
            self.camera_lost.emit("未安装 OpenCV，摄像头功能不可用", "设备: -")
            return
        self.cap, self.backend_name, errors = _open_video_capture(self.camera_index)
        if not self.cap.isOpened():
            msg = _build_camera_unavailable_message(self.camera_index, errors)
            detail = _build_camera_unavailable_detail(self.camera_index, errors)
            self.camera_lost.emit(msg, detail)
            return

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                detail = f"设备: camera:{self.camera_index}  后端: {self.backend_name or '-'}"
                self.camera_lost.emit("摄像头读取失败", detail)
                break
            self.frame_ready.emit(frame)
            self.msleep(30)

        if self.cap:
            self.cap.release()
            self.cap = None

    def stop(self):
        self.running = False
        self.wait()


class CameraScanThread(QThread):
    scan_finished = pyqtSignal(object)

    def __init__(self, max_index=8, skip_indices=None):
        super().__init__()
        self.max_index = int(max_index)
        self.skip_indices = list(skip_indices or [])

    def run(self):
        self.scan_finished.emit(scan_available_cameras(self.max_index, self.skip_indices))


class CameraPanel(QWidget):
    frame_captured = pyqtSignal(object)
    camera_error = pyqtSignal(str, int)

    def __init__(self, parent=None, camera_index=0):
        super().__init__(parent)

        self.label = QLabel("等待摄像头...")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(640, 360)
        self.label.setStyleSheet(
            """
            QLabel {
                background: #0f172a;
                color: #cbd5e1;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
            }
            """
        )

        lay = QVBoxLayout(self)
        lay.addWidget(self.label)

        self.thread = None
        self.current_camera_index = int(camera_index)
        self.source_kind = "local"
        self.no_camera_timer = QTimer(self)
        self.no_camera_timer.setInterval(8000)  # 每8秒重试并提示一次
        self.no_camera_timer.timeout.connect(self.try_reconnect)

        self.last_warn_time = 0
        self.last_warn_msg = None
        self.warn_interval_sec = 8
        self.last_frame = None
        self.face_annotations = []
        self.status_label = None
        self.info_label = None

        self.start_camera()

    def bind_status_labels(self, status_label, info_label):
        self.status_label = status_label
        self.info_label = info_label

    def selected_camera_id(self):
        return self.current_camera_index

    def selected_device_info(self):
        if self.source_kind == "mobile":
            return "mobile-browser"
        return f"camera:{self.current_camera_index}"

    def start_camera(self, camera_index=None):
        if self.thread and self.thread.isRunning():
            return
        if camera_index is not None:
            self.current_camera_index = int(camera_index)
        self.source_kind = "local"
        self.last_frame = None
        self.label.setText(f"正在连接 camera:{self.current_camera_index} ...")
        self.thread = CameraThread(camera_index=self.current_camera_index)
        self.thread.frame_ready.connect(self.update_frame)
        self.thread.camera_lost.connect(self.on_camera_lost)
        self.thread.start()

    def stop_camera(self):
        if self.no_camera_timer.isActive():
            self.no_camera_timer.stop()
        if self.thread is not None:
            try:
                self.thread.stop()
            finally:
                self.thread = None

    def switch_camera(self, camera_index):
        self.stop_camera()
        self.current_camera_index = int(camera_index)
        self.source_kind = "local"
        self.last_frame = None
        self.set_face_annotations([])
        self.label.clear()
        self.start_camera(self.current_camera_index)

    def use_mobile_source(self):
        self.stop_camera()
        self.source_kind = "mobile"
        self.last_frame = None
        self.set_face_annotations([])
        self.label.setText("等待手机浏览器画面...")
        if self.status_label is not None:
            self.status_label.setText("状态: 等待手机扫码")
        if self.info_label is not None:
            self.info_label.setText("设备: mobile-browser")

    def update_mobile_frame(self, frame_bgr):
        self.source_kind = "mobile"
        self.last_frame = frame_bgr
        self.frame_captured.emit(frame_bgr)
        if self.status_label is not None:
            self.status_label.setText("状态: 手机已连接")
        if self.info_label is not None:
            self.info_label.setText("设备: mobile-browser")
        self._render_frame(frame_bgr)

    def on_camera_lost(self, msg, detail):
        self.set_face_annotations([])
        self.camera_error.emit(msg, self.current_camera_index)
        now = time.time()
        should_warn = msg != self.last_warn_msg or now - self.last_warn_time >= self.warn_interval_sec
        if should_warn:
            QMessageBox.warning(self, "摄像头状态", msg)
            self.last_warn_time = now
            self.last_warn_msg = msg

        self.label.setText(msg)
        if self.status_label is not None:
            self.status_label.setText("状态: 不可用")
        if self.info_label is not None:
            self.info_label.setText(detail)
        if not self.no_camera_timer.isActive():
            self.no_camera_timer.start()

    def try_reconnect(self):
        if self.thread and self.thread.isRunning():
            return
        self.start_camera(self.current_camera_index)

    def update_frame(self, frame_bgr):
        if cv2 is None:
            self.label.setText("OpenCV 不可用，无法渲染视频")
            return
        if self.source_kind != "local":
            return

        if self.no_camera_timer.isActive():
            self.no_camera_timer.stop()

        self.last_frame = frame_bgr
        self.frame_captured.emit(frame_bgr)
        if self.status_label is not None:
            self.status_label.setText("状态: 已连接")
        if self.info_label is not None and self.thread is not None:
            backend = self.thread.backend_name or "-"
            self.info_label.setText(f"设备: camera:{self.current_camera_index}  后端: {backend}")

        self._render_frame(frame_bgr)

    def _render_frame(self, frame_bgr):
        if cv2 is None:
            self.label.setText("OpenCV 不可用，无法渲染视频")
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self._draw_annotations(qimg)
        pix = QPixmap.fromImage(qimg).scaled(
            self.label.width(), self.label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label.setPixmap(pix)

    def set_face_annotations(self, annotations):
        self.face_annotations = list(annotations or [])

    def scan_devices(self, max_index=8):
        skip = [self.current_camera_index] if self.source_kind == "local" and self.last_frame is not None else []
        devices = scan_available_cameras(max_index=max_index, skip_indices=skip)
        current_seen = any(int(d["index"]) == self.current_camera_index for d in devices)
        if self.source_kind == "local" and not current_seen and self.last_frame is not None:
            devices.insert(0, {
                "index": self.current_camera_index,
                "name": f"camera:{self.current_camera_index}",
            })
        return devices

    def _draw_annotations(self, qimg):
        if not self.face_annotations:
            return

        painter = QPainter(qimg)
        painter.setRenderHint(QPainter.Antialiasing, True)
        font = QFont("Microsoft YaHei", 14)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()

        for item in self.face_annotations:
            box = item.get("box") or []
            if len(box) < 4:
                continue

            x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
            x1 = max(0, min(x1, qimg.width() - 1))
            y1 = max(0, min(y1, qimg.height() - 1))
            x2 = max(0, min(x2, qimg.width() - 1))
            y2 = max(0, min(y2, qimg.height() - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            matched = bool(item.get("matched"))
            stroke = QColor(34, 197, 94) if matched else QColor(245, 158, 11)
            painter.setPen(QPen(stroke, 3))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))

            username = item.get("username") or "陌生人"
            similarity = float(item.get("similarity") or 0.0)
            text = f"{username} {similarity:.3f}"
            padding_x = 8
            padding_y = 5
            text_w = metrics.horizontalAdvance(text)
            text_h = metrics.height()
            label_w = min(qimg.width(), text_w + padding_x * 2)
            label_h = text_h + padding_y * 2
            label_x = min(max(0, x1), max(0, qimg.width() - label_w))
            label_y = max(0, y1 - label_h)
            if label_y == 0:
                label_y = min(max(0, qimg.height() - label_h), y1)

            bg = QColor(stroke)
            bg.setAlpha(230)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(bg))
            painter.drawRect(QRect(label_x, label_y, label_w, label_h))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                QRect(label_x + padding_x, label_y + padding_y, max(1, label_w - padding_x * 2), text_h),
                Qt.AlignLeft | Qt.AlignVCenter,
                text,
            )

        painter.end()

    def captureFrame(self, camera_id=None, resolution="640x480", frame_rate=30):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.source_kind == "mobile":
            if self.last_frame is None:
                return {
                    "code": 5006,
                    "success": False,
                    "message": "尚未收到手机浏览器画面",
                    "data": {"frame_data": None, "device_info": "mobile-browser"},
                    "timestamp": now_str,
                }
            return {
                "code": 0,
                "success": True,
                "message": "采集成功",
                "data": {
                    "frame_data": self.last_frame,
                    "camera_id": "mobile-browser",
                    "device_info": "mobile-browser",
                    "resolution": resolution,
                    "frame_rate": frame_rate,
                },
                "timestamp": now_str,
            }

        target_camera_id = self.current_camera_index if camera_id is None else int(camera_id)
        if self.last_frame is not None and target_camera_id == self.current_camera_index:
            return {
                "code": 0,
                "success": True,
                "message": "采集成功",
                "data": {
                    "frame_data": self.last_frame,
                    "camera_id": target_camera_id,
                    "device_info": f"camera:{target_camera_id}",
                    "resolution": resolution,
                    "frame_rate": frame_rate,
                },
                "timestamp": now_str,
            }

        if cv2 is None:
            return {
                "code": 5003,
                "success": False,
                "message": "未安装 OpenCV，无法采集图像",
                "data": {"frame_data": None, "device_info": f"camera:{target_camera_id}"},
                "timestamp": now_str,
            }

        cap, backend_name, errors = _open_video_capture(target_camera_id)
        if not cap.isOpened():
            return {
                "code": 5001,
                "success": False,
                "message": _build_camera_unavailable_message(target_camera_id, errors),
                "data": {
                    "frame_data": None,
                    "device_info": f"camera:{target_camera_id}",
                    "backend": None,
                },
                "timestamp": now_str,
            }

        width, height = 640, 480
        try:
            w_str, h_str = resolution.lower().split("x")
            width, height = int(w_str), int(h_str)
        except Exception:
            pass

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, frame_rate)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return {
                "code": 5002,
                "success": False,
                "message": "采集超时或读取失败",
                "data": {"frame_data": None, "device_info": f"camera:{target_camera_id}"},
                "timestamp": now_str,
            }

        if target_camera_id == self.current_camera_index:
            self.last_frame = frame
        return {
            "code": 0,
            "success": True,
            "message": "采集成功",
            "data": {
                "frame_data": frame,
                "camera_id": target_camera_id,
                "device_info": f"camera:{target_camera_id}",
                "resolution": f"{width}x{height}",
                "frame_rate": frame_rate,
                "backend": backend_name,
            },
            "timestamp": now_str,
        }

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)
