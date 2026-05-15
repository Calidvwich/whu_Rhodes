from __future__ import annotations

import ipaddress
import ssl
import socket
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


CERT_DIR = Path(__file__).resolve().parent / ".mobile_camera"
CERT_PATH = CERT_DIR / "cert.pem"
KEY_PATH = CERT_DIR / "key.pem"


MOBILE_CAMERA_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>手机摄像头</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: system-ui, -apple-system, "Microsoft YaHei", sans-serif;
      background: #0f172a;
      color: #f8fafc;
    }
    main {
      width: min(760px, 100%);
      margin: 0 auto;
      padding: 18px;
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 14px;
      margin-bottom: 14px;
    }
    h2 { margin: 0 0 6px; font-size: 24px; }
    .hint { margin: 0; color: #cbd5e1; line-height: 1.55; font-size: 14px; }
    .badge {
      min-width: 76px;
      text-align: center;
      padding: 7px 10px;
      border-radius: 999px;
      background: #1e293b;
      color: #93c5fd;
      font-size: 13px;
      font-weight: 700;
      border: 1px solid #334155;
    }
    .notice {
      margin: 0 0 14px;
      padding: 10px 12px;
      border-radius: 10px;
      background: #172554;
      border: 1px solid #2563eb;
      color: #dbeafe;
      font-weight: 700;
      line-height: 1.45;
    }
    .preview {
      position: relative;
      overflow: hidden;
      border-radius: 14px;
      background: #000;
      border: 1px solid #334155;
      box-shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
    }
    #previewCanvas {
      display: block;
      width: 100%;
      min-height: 320px;
      max-height: 64vh;
      background: #000;
    }
    .controls {
      margin-top: 14px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    button {
      min-height: 48px;
      border: 0;
      border-radius: 10px;
      padding: 0 14px;
      font-size: 15px;
      font-weight: 800;
      color: #fff;
      background: #4c8bf5;
      box-shadow: 0 8px 20px rgba(76, 139, 245, 0.24);
    }
    button.secondary {
      background: #1e293b;
      color: #e2e8f0;
      box-shadow: none;
      border: 1px solid #334155;
    }
    button.danger {
      background: #ef4444;
      box-shadow: 0 8px 20px rgba(239, 68, 68, 0.18);
    }
    #startBtn { grid-column: span 2; }
    #status {
      margin-top: 12px;
      padding: 12px 13px;
      min-height: 44px;
      border-radius: 10px;
      background: #111827;
      border: 1px solid #243244;
      color: #93c5fd;
      line-height: 1.5;
      font-size: 14px;
    }
  </style>
</head>
<body>
  <main>
    <div class="topbar">
      <div>
        <h2>手机摄像头</h2>
        <p class="hint">允许浏览器访问摄像头后，画面会发送到电脑端用于人脸识别。</p>
      </div>
      <div id="modeBadge" class="badge">前置 · 竖屏</div>
    </div>
    <p class="notice">请勿关闭此界面；关闭或锁屏后电脑端将无法继续接收手机摄像头画面。</p>
    <div class="preview">
      <canvas id="previewCanvas"></canvas>
    </div>
    <video id="video" autoplay playsinline muted style="position:absolute;width:1px;height:1px;opacity:0;pointer-events:none;"></video>
    <canvas id="canvas" style="display:none"></canvas>
    <div class="controls">
      <button id="startBtn">开始发送</button>
      <button id="switchBtn" class="secondary">切换后置</button>
      <button id="orientationBtn" class="secondary">切换横屏</button>
      <button id="stopBtn" class="danger">停止</button>
    </div>
    <div id="status">等待启动</div>
  </main>
  <script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const previewCanvas = document.getElementById('previewCanvas');
    const statusEl = document.getElementById('status');
    const modeBadge = document.getElementById('modeBadge');
    const startBtn = document.getElementById('startBtn');
    const switchBtn = document.getElementById('switchBtn');
    const orientationBtn = document.getElementById('orientationBtn');
    const stopBtn = document.getElementById('stopBtn');
    let stream = null;
    let timer = null;
    let busy = false;
    let facingMode = 'user';
    let orientation = 'portrait';

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function stopStreamOnly() {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
      }
      video.srcObject = null;
    }

    function updateSwitchButton() {
      switchBtn.textContent = facingMode === 'user' ? '切换后置' : '切换前置';
    }

    function updateOrientationButton() {
      orientationBtn.textContent = orientation === 'portrait' ? '切换横屏' : '切换竖屏';
    }

    function updateBadge() {
      modeBadge.textContent = (facingMode === 'user' ? '前置' : '后置') + ' · ' + (orientation === 'portrait' ? '竖屏' : '横屏');
    }

    function syncControls() {
      updateSwitchButton();
      updateOrientationButton();
      updateBadge();
    }

    function landscapeRotation() {
      return facingMode === 'environment' ? -Math.PI / 2 : Math.PI / 2;
    }

    function drawOrientedFrame(targetCanvas, maxWidth) {
      const scale = Math.min(1, maxWidth / video.videoWidth);
      const srcWidth = Math.round(video.videoWidth * scale);
      const srcHeight = Math.round(video.videoHeight * scale);
      const ctx = targetCanvas.getContext('2d');

      if (orientation === 'landscape') {
        targetCanvas.width = srcHeight;
        targetCanvas.height = srcWidth;
        ctx.save();
        ctx.translate(targetCanvas.width / 2, targetCanvas.height / 2);
        ctx.rotate(landscapeRotation());
        ctx.drawImage(video, -srcWidth / 2, -srcHeight / 2, srcWidth, srcHeight);
        ctx.restore();
      } else {
        targetCanvas.width = srcWidth;
        targetCanvas.height = srcHeight;
        ctx.drawImage(video, 0, 0, targetCanvas.width, targetCanvas.height);
      }
    }

    async function openCamera() {
      const previousStream = stream;
      try {
        if (previousStream) {
          previousStream.getTracks().forEach(track => track.stop());
        }
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: facingMode },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
        video.srcObject = stream;
        clearInterval(timer);
        timer = setInterval(sendFrame, 200);
        syncControls();
        setStatus((facingMode === 'user' ? '前置' : '后置') + '摄像头已授权，正在发送画面');
      } catch (err) {
        stream = null;
        video.srcObject = null;
        clearInterval(timer);
        timer = null;
        setStatus('摄像头授权或切换失败：' + err.message);
        throw err;
      }
    }

    async function sendFrame() {
      if (!stream || busy || video.videoWidth === 0 || video.videoHeight === 0) return;
      busy = true;
      drawOrientedFrame(previewCanvas, 960);
      drawOrientedFrame(canvas, 640);
      canvas.toBlob(async (blob) => {
        try {
          await fetch('/frame', {
            method: 'POST',
            headers: {'Content-Type': 'image/jpeg'},
            body: blob,
            cache: 'no-store',
          });
          setStatus('正在发送画面到电脑');
        } catch (err) {
          setStatus('发送失败：请确认电脑端服务仍在运行');
        } finally {
          busy = false;
        }
      }, 'image/jpeg', 0.82);
    }

    startBtn.onclick = async () => {
      try {
        facingMode = 'user';
        await openCamera();
      } catch (err) {
        setStatus('摄像头授权失败：' + err.message);
      }
    };

    switchBtn.onclick = async () => {
      const oldFacingMode = facingMode;
      facingMode = facingMode === 'user' ? 'environment' : 'user';
      try {
        await openCamera();
      } catch (err) {
        facingMode = oldFacingMode;
        syncControls();
      }
    };

    orientationBtn.onclick = () => {
      orientation = orientation === 'portrait' ? 'landscape' : 'portrait';
      syncControls();
      setStatus((orientation === 'portrait' ? '竖屏' : '横屏') + '发送模式已启用');
    };

    stopBtn.onclick = () => {
      clearInterval(timer);
      timer = null;
      stopStreamOnly();
      setStatus('已停止');
    };

    syncControls();
  </script>
</body>
</html>
"""


def get_lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        sock.close()


def ensure_self_signed_cert(host_ip: str) -> tuple[Path, Path]:
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "whu_Rhodes"),
        x509.NameAttribute(NameOID.COMMON_NAME, "whu-rhodes-mobile-camera"),
    ])

    alt_names: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
    ]
    try:
        alt_names.append(x509.IPAddress(ipaddress.ip_address(host_ip)))
    except ValueError:
        alt_names.append(x509.DNSName(host_ip))

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow() - timedelta(days=1))
        .not_valid_after(datetime.utcnow() + timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .sign(key, hashes.SHA256())
    )

    KEY_PATH.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    CERT_PATH.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return CERT_PATH, KEY_PATH


class MobileCameraHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, handler_class, frame_callback):
        super().__init__(server_address, handler_class)
        self.frame_callback = frame_callback


class MobileCameraRequestHandler(BaseHTTPRequestHandler):
    server: MobileCameraHTTPServer

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_text("OK", content_type="text/plain")
            return
        if parsed.path in {"", "/"}:
            self._send_text(MOBILE_CAMERA_HTML, content_type="text/html; charset=utf-8")
            return
        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/frame":
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0 or content_length > 2 * 1024 * 1024:
            self.send_error(400)
            return

        frame_bytes = self.rfile.read(content_length)
        try:
            self.server.frame_callback(frame_bytes)
        except Exception:
            self.send_error(500)
            return

        self.send_response(204)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def log_message(self, _format, *_args):
        return

    def _send_text(self, text: str, *, content_type: str):
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


class MobileCameraServer:
    def __init__(self, frame_callback, *, port: int = 8765) -> None:
        self.frame_callback = frame_callback
        self.port = int(port)
        self.host_ip = get_lan_ip()
        self.httpd: MobileCameraHTTPServer | None = None
        self.thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"https://{self.host_ip}:{self.port}/"

    def start(self) -> str:
        if self.httpd is not None:
            return self.url

        self.host_ip = get_lan_ip()
        cert_path, key_path = ensure_self_signed_cert(self.host_ip)
        httpd = MobileCameraHTTPServer(("0.0.0.0", self.port), MobileCameraRequestHandler, self.frame_callback)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=cert_path, keyfile=key_path)
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

        thread = threading.Thread(target=httpd.serve_forever, name="MobileCameraServer", daemon=True)
        thread.start()

        self.httpd = httpd
        self.thread = thread
        return self.url

    def stop(self) -> None:
        if self.httpd is None:
            return

        httpd = self.httpd
        self.httpd = None
        httpd.shutdown()
        httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
