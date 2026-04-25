import importlib
import time
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QMessageBox


def _optional_import(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


cv2 = _optional_import("cv2")


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
            self.camera_lost.emit("未安装 OpenCV，摄像头功能不可用")
            return
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.camera_lost.emit("未检测到摄像头")
            return

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.camera_lost.emit("摄像头读取失败")
                break
            self.frame_ready.emit(frame)
            self.msleep(30)

        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False
        self.wait()


class CameraPanel(QWidget):
    frame_captured = pyqtSignal(object)

    def __init__(self, parent=None):
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

        self.last_warn_time = 0
        self.warn_interval_sec = 8
        self.last_frame = None

        self.start_camera()

    def start_camera(self):
        if self.thread and self.thread.isRunning():
            return
        self.thread = CameraThread(camera_index=0)
        self.thread.frame_ready.connect(self.update_frame)
        self.thread.camera_lost.connect(self.on_camera_lost)
        self.thread.start()

    def on_camera_lost(self, msg):
        now = time.time()
        if now - self.last_warn_time >= self.warn_interval_sec:
            QMessageBox.warning(self, "摄像头状态", f"{msg}，请检查设备连接。")
            self.last_warn_time = now

        self.label.setText("未检测到摄像头，正在周期性重试...")
        if not self.no_camera_timer.isActive():
            self.no_camera_timer.start()

    def try_reconnect(self):
        if self.thread and self.thread.isRunning():
            return
        self.start_camera()

    def update_frame(self, frame_bgr):
        if cv2 is None:
            self.label.setText("OpenCV 不可用，无法渲染视频")
            return

        if self.no_camera_timer.isActive():
            self.no_camera_timer.stop()

        self.last_frame = frame_bgr
        self.frame_captured.emit(frame_bgr)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.label.width(), self.label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label.setPixmap(pix)

    def captureFrame(self, camera_id=0, resolution="640x480", frame_rate=30):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.last_frame is not None:
            return {
                "code": 0,
                "success": True,
                "message": "采集成功",
                "data": {
                    "frame_data": self.last_frame,
                    "camera_id": camera_id,
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
                "data": {"frame_data": None},
                "timestamp": now_str,
            }

        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return {
                "code": 5001,
                "success": False,
                "message": "摄像头不可用",
                "data": {"frame_data": None},
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
                "data": {"frame_data": None},
                "timestamp": now_str,
            }

        self.last_frame = frame
        return {
            "code": 0,
            "success": True,
            "message": "采集成功",
            "data": {
                "frame_data": frame,
                "camera_id": camera_id,
                "resolution": f"{width}x{height}",
                "frame_rate": frame_rate,
            },
            "timestamp": now_str,
        }

    def closeEvent(self, event):
        if self.thread:
            self.thread.stop()
        super().closeEvent(event)