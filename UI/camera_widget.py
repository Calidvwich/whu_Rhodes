import importlib
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout


def _optional_import(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


cv2 = _optional_import("cv2")


def camera_display_name(camera_index):
    if int(camera_index) == 0:
        return "电脑摄像头 camera:0"
    return f"可用摄像头 camera:{int(camera_index)}"


def scan_available_cameras(max_index=8, skip_indices=None):
    devices = []
    if cv2 is None:
        return devices

    skip = {int(i) for i in (skip_indices or [])}
    for index in range(int(max_index) + 1):
        if index in skip:
            continue
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        try:
            if cap.isOpened():
                devices.append({
                    "index": index,
                    "name": camera_display_name(index),
                })
        finally:
            cap.release()
    return devices


class CameraThread(QThread):
    frame_ready = pyqtSignal(object)   # numpy frame (BGR)
    camera_lost = pyqtSignal(str)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.cap = None

    def run(self):
        self.running = True
        if cv2 is None:
            self.camera_lost.emit(f"camera:{self.camera_index} 未安装 OpenCV，摄像头功能不可用")
            return
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.camera_lost.emit(f"camera:{self.camera_index} 未检测到摄像头")
            self.cap.release()
            self.cap = None
            return

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.camera_lost.emit(f"camera:{self.camera_index} 摄像头读取失败")
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

        lay = QVBoxLayout(self)
        lay.addWidget(self.label)

        self.thread = None
        self.no_camera_timer = QTimer(self)
        self.no_camera_timer.setInterval(8000)  # 每8秒重试并提示一次
        self.no_camera_timer.timeout.connect(self.try_reconnect)

        self.last_frame = None
        self.current_camera_index = int(camera_index)
        self.source_kind = "local"

        self.start_camera()

    def start_camera(self, camera_index=None):
        if camera_index is not None:
            self.current_camera_index = int(camera_index)
        if self.thread and self.thread.isRunning():
            return
        self.source_kind = "local"
        self.label.setText(f"正在连接 camera:{self.current_camera_index} ...")
        self.thread = CameraThread(camera_index=self.current_camera_index)
        self.thread.frame_ready.connect(self.update_frame)
        self.thread.camera_lost.connect(self.on_camera_lost)
        self.thread.start()

    def stop_camera(self):
        if self.thread:
            self.thread.stop()
            self.thread = None

    def switch_camera(self, camera_index):
        camera_index = int(camera_index)
        if camera_index == self.current_camera_index and self.thread and self.thread.isRunning():
            return True

        if self.no_camera_timer.isActive():
            self.no_camera_timer.stop()
        self.stop_camera()
        self.current_camera_index = camera_index
        self.source_kind = "local"
        self.last_frame = None
        self.label.clear()
        self.start_camera(camera_index)
        return True

    def scan_devices(self, max_index=8):
        skip = [self.current_camera_index] if self.source_kind == "local" and self.last_frame is not None else []
        devices = scan_available_cameras(max_index=max_index, skip_indices=skip)
        current_seen = any(d["index"] == self.current_camera_index for d in devices)
        if self.source_kind == "local" and not current_seen and self.last_frame is not None:
            devices.insert(0, {
                "index": self.current_camera_index,
                "name": camera_display_name(self.current_camera_index),
            })
        return devices

    def selected_camera_id(self):
        return self.current_camera_index

    def selected_device_info(self):
        if self.source_kind == "mobile":
            return "mobile-browser"
        return f"camera:{self.current_camera_index}"

    def use_mobile_source(self):
        if self.no_camera_timer.isActive():
            self.no_camera_timer.stop()
        self.stop_camera()
        self.source_kind = "mobile"
        self.last_frame = None
        self.label.setText("等待手机浏览器画面...")

    def update_mobile_frame(self, frame_bgr):
        if self.source_kind != "mobile":
            self.use_mobile_source()
        self.update_frame(frame_bgr)

    def on_camera_lost(self, msg):
        self.camera_error.emit(msg, self.current_camera_index)
        self.label.setText("未检测到摄像头，点击刷新或等待周期性重试...")
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

        if self.no_camera_timer.isActive():
            self.no_camera_timer.stop()

        self.last_frame = frame_bgr.copy()
        self.frame_captured.emit(self.last_frame)

        rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            self.label.width(), self.label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label.setPixmap(pix)

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

        cap = cv2.VideoCapture(target_camera_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return {
                "code": 5001,
                "success": False,
                "message": f"camera:{target_camera_id} 摄像头不可用",
                "data": {"frame_data": None, "device_info": f"camera:{target_camera_id}"},
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
            },
            "timestamp": now_str,
        }

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)
