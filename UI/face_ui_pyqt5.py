import sys
import math
import os
import hashlib
import importlib
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from camera_widget import CameraPanel


def _optional_import(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


cv2 = _optional_import("cv2")
np = _optional_import("numpy")
bcrypt_mod = _optional_import("bcrypt")


# ========= 全局样式 =========
QSS = """
QMainWindow, QWidget {
    background-color: #f5f7fb;
    font-family: "Microsoft YaHei";
    font-size: 14px;
}
#titleLabel {
    font-size: 28px;
    font-weight: 700;
    color: #1f2d3d;
}
#card {
    background: white;
    border: 1px solid #e6ebf2;
    border-radius: 14px;
}
#subtitle {
    font-size: 20px;
    font-weight: 600;
    color: #2c3e50;
}
QLineEdit {
    height: 38px;
    border: 1px solid #d7deea;
    border-radius: 8px;
    padding: 0 10px;
    background: #fcfdff;
}
QLineEdit:focus {
    border: 1px solid #4c8bf5;
}
QPushButton {
    height: 38px;
    border: none;
    border-radius: 8px;
    padding: 0 16px;
    background-color: #4c8bf5;
    color: white;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #3f7ee8;
}
QPushButton:pressed {
    background-color: #326fdf;
}
#panelTitle {
    font-size: 16px;
    font-weight: 600;
    color: #2c3e50;
}
#placeholder {
    color: #7f8c9f;
}
"""


def build_response(code, success, message, data=None):
    return {
        "code": code,
        "success": success,
        "message": message,
        "data": data or {},
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ========= 数据持久层 =========
class UserRepository:
    def __init__(self):
        self._users = {}
        self._user_id_seq = 1000

    def getUserByUsername(self, username):
        return self._users.get(username)

    def getUserById(self, user_id):
        for item in self._users.values():
            if item["user_id"] == user_id:
                return item
        return None

    def hasDuplicateAccount(self, username, phone, email):
        if username in self._users:
            return True
        for user in self._users.values():
            if phone and user.get("phone") == phone:
                return True
            if email and user.get("email") == email:
                return True
        return False

    def insertUser(self, username, password_hash, phone, email, role="user"):
        self._user_id_seq += 1
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "user_id": self._user_id_seq,
            "username": username,
            "password": password_hash,
            "phone": phone,
            "email": email,
            "create_time": now_str,
            "status": "enabled",
            "role": role,
            "photo": "",
        }
        self._users[username] = record
        return record

    def updateUserStatus(self, user_id, status):
        user = self.getUserById(user_id)
        if not user:
            return False
        user["status"] = status
        return True

    def updateUserInfo(self, old_username, new_username, password_hash=None, photo=None):
        old = self._users.get(old_username)
        if not old:
            return False, "原用户名不存在"
        if new_username != old_username and new_username in self._users:
            return False, "新用户名已存在"

        self._users.pop(old_username)
        old["username"] = new_username
        if password_hash:
            old["password"] = password_hash
        if photo is not None:
            old["photo"] = photo
        self._users[new_username] = old
        return True, "用户信息修改成功"

    def deleteUser(self, username):
        if username not in self._users:
            return False
        self._users.pop(username)
        return True


class FaceFeatureRepository:
    def __init__(self):
        self._features = []
        self._feature_id_seq = 0

    def insertFaceFeature(self, user_id, feature_vector, image_path):
        self._feature_id_seq += 1
        if np is not None:
            vec = np.asarray(feature_vector, dtype=np.float32)
        else:
            vec = [float(x) for x in feature_vector]
        item = {
            "feature_id": self._feature_id_seq,
            "user_id": user_id,
            "feature_vector": vec,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_active": True,
            "image_path": image_path,
        }
        self._features.append(item)
        return item

    def getFaceFeaturesByUserId(self, user_id):
        return [f for f in self._features if f["user_id"] == user_id and f["is_active"]]

    def getAllActiveFeatures(self):
        return [f for f in self._features if f["is_active"]]


class RecognitionLogRepository:
    def __init__(self):
        self._logs = []
        self._log_id_seq = 0

    def insertRecognitionLog(self, user_id, capture_image, similarity, result, recognize_time, device_info):
        self._log_id_seq += 1
        row = {
            "log_id": self._log_id_seq,
            "user_id": user_id,
            "capture_image": capture_image,
            "similarity": float(similarity),
            "result": result,
            "recognize_time": recognize_time,
            "device_info": device_info,
        }
        self._logs.append(row)
        return row

    def getRecognitionLogsByUserId(self, user_id):
        return [x for x in self._logs if x["user_id"] == user_id]

    def getRecognitionLogsByTimeRange(self, start_time, end_time):
        result = []
        for row in self._logs:
            t = datetime.strptime(row["recognize_time"], "%Y-%m-%d %H:%M:%S")
            if start_time <= t <= end_time:
                result.append(row)
        return result


class SystemConfigRepository:
    def __init__(self):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._configs = {
            "recognition_threshold": {
                "config_id": 1,
                "config_key": "recognition_threshold",
                "config_value": "0.82",
                "update_time": now_str,
                "description": "识别阈值",
            },
            "model_version": {
                "config_id": 2,
                "config_key": "model_version",
                "config_value": "demo-v1",
                "update_time": now_str,
                "description": "模型版本",
            },
            "oss_path": {
                "config_id": 3,
                "config_key": "oss_path",
                "config_value": "local://face_data",
                "update_time": now_str,
                "description": "对象存储路径",
            },
        }

    def getConfigByKey(self, config_key):
        return self._configs.get(config_key)

    def updateConfig(self, config_key, config_value):
        if config_key not in self._configs:
            return False
        self._configs[config_key]["config_value"] = str(config_value)
        self._configs[config_key]["update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return True


# ========= 算法支撑层 =========
class FaceAlgorithm:
    def __init__(self):
        self.detector = None
        if cv2 is not None:
            cascade_name = "haarcascade_frontalface_default.xml"
            candidate_paths = []

            cv2_data = getattr(cv2, "data", None)
            if cv2_data is not None:
                haar_dir = getattr(cv2_data, "haarcascades", None)
                if haar_dir:
                    candidate_paths.append(os.path.join(haar_dir, cascade_name))

            cv2_file = getattr(cv2, "__file__", "")
            if cv2_file:
                cv2_dir = os.path.dirname(cv2_file)
                candidate_paths.extend([
                    os.path.join(cv2_dir, "data", cascade_name),
                    os.path.join(cv2_dir, "..", "share", "opencv4", "haarcascades", cascade_name),
                    os.path.join(cv2_dir, "..", "etc", "haarcascades", cascade_name),
                ])

            for path in candidate_paths:
                normalized_path = os.path.normpath(path)
                if not os.path.exists(normalized_path):
                    continue
                detector = cv2.CascadeClassifier(normalized_path)
                if detector is not None and not detector.empty():
                    self.detector = detector
                    break

    def detectFace(self, image_data):
        if image_data is None:
            return build_response(4100, False, "图像为空", {"face_found": False, "face_region": []})

        if cv2 is None:
            return build_response(4102, False, "未安装 OpenCV，无法执行人脸检测", {
                "face_found": False,
                "face_region": [],
            })

        if self.detector is None or self.detector.empty():
            return build_response(4103, False, "OpenCV 已安装，但未找到可用的人脸检测模型", {
                "face_found": False,
                "face_region": [],
            })

        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        face_region = [[int(x), int(y), int(w), int(h)] for x, y, w, h in faces]
        found = len(face_region) > 0
        message = "检测到人脸" if found else "未检测到人脸"
        return build_response(0 if found else 4101, found, message, {
            "face_found": found,
            "face_region": face_region,
        })

    def extractFeature(self, face_image):
        if face_image is None or getattr(face_image, "size", 0) == 0:
            return build_response(4200, False, "无效的人脸区域", {
                "feature_vector": [],
            })

        if cv2 is None or np is None:
            raw = b""
            try:
                raw = face_image.tobytes()
            except Exception:
                raw = b""
            if not raw:
                raw = hashlib.sha256(str(face_image).encode("utf-8")).digest()
            repeated = (raw * ((512 // len(raw)) + 1))[:512]
            feature_vector = [b / 255.0 for b in repeated]
            return build_response(0, True, "特征提取成功（简化模式）", {
                "feature_vector": feature_vector,
                "vector_dim": 512,
            })

        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32).flatten() / 255.0

        if normalized.size >= 512:
            feature_vector = normalized[:512]
        else:
            repeat_count = int(np.ceil(512 / normalized.size))
            feature_vector = np.tile(normalized, repeat_count)[:512]

        return build_response(0, True, "特征提取成功", {
            "feature_vector": feature_vector,
            "vector_dim": 512,
        })

    def compareFeature(self, input_vector, feature_library, threshold):
        if input_vector is None or len(input_vector) == 0:
            return build_response(4300, False, "输入特征为空", {
                "matched_user_id": None,
                "max_similarity": 0.0,
                "matched": False,
            })

        if not feature_library:
            return build_response(4301, True, "特征库为空，判定为陌生人", {
                "matched_user_id": None,
                "max_similarity": 0.0,
                "matched": False,
            })

        input_vec = [float(x) for x in input_vector]
        input_norm = math.sqrt(sum(x * x for x in input_vec))
        best_score = -1.0
        best_user_id = None

        for feature in feature_library:
            vec = [float(x) for x in feature["feature_vector"]]
            vec_norm = math.sqrt(sum(x * x for x in vec))
            denom = input_norm * vec_norm
            if denom == 0:
                continue
            sim = float(sum(a * b for a, b in zip(input_vec, vec)) / denom)
            if sim > best_score:
                best_score = sim
                best_user_id = feature["user_id"]

        if best_score < 0:
            best_score = 0.0
        matched = best_score >= float(threshold)

        return build_response(0, True, "比对完成", {
            "matched_user_id": best_user_id if matched else None,
            "max_similarity": best_score,
            "matched": matched,
        })


# ========= 业务逻辑层 =========
class FaceBusinessService:
    def __init__(self):
        self.user_repo = UserRepository()
        self.face_repo = FaceFeatureRepository()
        self.log_repo = RecognitionLogRepository()
        self.config_repo = SystemConfigRepository()
        self.algorithm = FaceAlgorithm()
        self._seed_demo_users()

    def _hash_password(self, password):
        if bcrypt_mod:
            return bcrypt_mod.hashpw(password.encode("utf-8"), bcrypt_mod.gensalt()).decode("utf-8")
        return "sha256$" + hashlib.sha256(password.encode("utf-8")).hexdigest()

    def _verify_password(self, password, password_hash):
        if password_hash.startswith("sha256$"):
            target = "sha256$" + hashlib.sha256(password.encode("utf-8")).hexdigest()
            return target == password_hash
        if bcrypt_mod:
            return bcrypt_mod.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
        return False

    def _seed_demo_users(self):
        if self.user_repo.getUserByUsername("admin"):
            return
        self.user_repo.insertUser(
            username="admin",
            password_hash=self._hash_password("admin123"),
            phone="13800000000",
            email="admin@example.com",
            role="admin",
        )
        self.user_repo.insertUser(
            username="user1",
            password_hash=self._hash_password("123456"),
            phone="13900000000",
            email="user1@example.com",
            role="user",
        )

    def getUserByUsername(self, username):
        user = self.user_repo.getUserByUsername(username)
        if not user:
            return build_response(4040, False, "用户不存在", {})
        return build_response(0, True, "查询成功", {"user": user})

    def login(self, username, password):
        user = self.user_repo.getUserByUsername(username)
        if not user:
            return build_response(4011, False, "用户未注册", {
                "user_id": None,
                "role": None,
            })

        if user["status"] != "enabled":
            return build_response(4013, False, "账号被禁用", {
                "user_id": user["user_id"],
                "role": user["role"],
            })

        if not self._verify_password(password, user["password"]):
            return build_response(4012, False, "密码错误", {
                "user_id": user["user_id"],
                "role": user["role"],
            })

        return build_response(0, True, "登录成功", {
            "user_id": user["user_id"],
            "role": user["role"],
            "username": user["username"],
        })

    def register(self, username, password, phone, email):
        if not username or not password:
            return build_response(4020, False, "用户名和密码不能为空", {"user_id": None})

        if self.user_repo.hasDuplicateAccount(username, phone, email):
            return build_response(4021, False, "账号已存在", {"user_id": None})

        password_hash = self._hash_password(password)
        user = self.user_repo.insertUser(username, password_hash, phone, email, role="user")
        return build_response(0, True, "注册成功", {"user_id": user["user_id"]})

    def addFaceFeature(self, user_id, image_file, operator_id):
        operator = self.user_repo.getUserById(operator_id)
        if not operator or operator["role"] != "admin" or operator["status"] != "enabled":
            return build_response(4031, False, "管理员权限不足", {"feature_id": None})

        user = self.user_repo.getUserById(user_id)
        if not user:
            return build_response(4032, False, "目标用户不存在", {"feature_id": None})

        image = self._to_image(image_file)
        if image is None:
            return build_response(4033, False, "图像格式不合法或路径无效", {"feature_id": None})

        detect_resp = self.algorithm.detectFace(image)
        if not detect_resp["success"]:
            return build_response(4034, False, "未检测到有效人脸", {"feature_id": None})

        x, y, w, h = detect_resp["data"]["face_region"][0]
        face_image = image[y:y + h, x:x + w]
        extract_resp = self.algorithm.extractFeature(face_image)
        if not extract_resp["success"]:
            return build_response(4035, False, "特征提取失败", {"feature_id": None})

        image_path = image_file if isinstance(image_file, str) else "memory_frame"
        row = self.face_repo.insertFaceFeature(
            user_id=user_id,
            feature_vector=extract_resp["data"]["feature_vector"],
            image_path=image_path,
        )
        return build_response(0, True, "录入成功", {"feature_id": row["feature_id"]})

    def recognizeFace(self, image_data, device_info, request_time):
        if image_data is None:
            return build_response(4041, False, "图像为空", {
                "user_id": None,
                "username": None,
                "similarity": 0.0,
                "result": "识别失败",
            })

        detect_resp = self.algorithm.detectFace(image_data)
        if not detect_resp["success"]:
            self.log_repo.insertRecognitionLog(
                user_id=None,
                capture_image="memory_frame",
                similarity=0.0,
                result="未检测到人脸",
                recognize_time=request_time,
                device_info=device_info,
            )
            return build_response(4042, False, "未检测到人脸", {
                "user_id": None,
                "username": None,
                "similarity": 0.0,
                "result": "未识别",
            })

        x, y, w, h = detect_resp["data"]["face_region"][0]
        face = image_data[y:y + h, x:x + w]
        extract_resp = self.algorithm.extractFeature(face)
        if not extract_resp["success"]:
            self.log_repo.insertRecognitionLog(
                user_id=None,
                capture_image="memory_frame",
                similarity=0.0,
                result="特征提取失败",
                recognize_time=request_time,
                device_info=device_info,
            )
            return build_response(4043, False, "特征提取失败", {
                "user_id": None,
                "username": None,
                "similarity": 0.0,
                "result": "识别失败",
            })

        threshold_row = self.config_repo.getConfigByKey("recognition_threshold")
        threshold = float(threshold_row["config_value"]) if threshold_row else 0.82
        compare_resp = self.algorithm.compareFeature(
            extract_resp["data"]["feature_vector"],
            self.face_repo.getAllActiveFeatures(),
            threshold,
        )

        matched = compare_resp["data"]["matched"]
        matched_user_id = compare_resp["data"]["matched_user_id"]
        similarity = float(compare_resp["data"]["max_similarity"])

        username = None
        result = "陌生人"
        if matched and matched_user_id is not None:
            user = self.user_repo.getUserById(matched_user_id)
            username = user["username"] if user else None
            result = "已识别"

        self.log_repo.insertRecognitionLog(
            user_id=matched_user_id if matched else None,
            capture_image="memory_frame",
            similarity=similarity,
            result=result,
            recognize_time=request_time,
            device_info=device_info,
        )

        return build_response(0, True, "识别完成", {
            "user_id": matched_user_id if matched else None,
            "username": username,
            "similarity": similarity,
            "result": result,
        })

    def updateConfig(self, operator_id, config_key, config_value):
        operator = self.user_repo.getUserById(operator_id)
        if not operator or operator["role"] != "admin":
            return build_response(4051, False, "仅管理员可修改配置", {})
        ok = self.config_repo.updateConfig(config_key, config_value)
        if not ok:
            return build_response(4052, False, "配置键不存在", {})
        return build_response(0, True, "配置更新成功", {
            "config_key": config_key,
            "config_value": str(config_value),
        })

    def updateUser(self, old_username, new_username, new_password, new_photo):
        password_hash = self._hash_password(new_password) if new_password else None
        ok, msg = self.user_repo.updateUserInfo(old_username, new_username, password_hash, new_photo)
        if not ok:
            return build_response(4061, False, msg, {})
        return build_response(0, True, msg, {})

    def deleteUser(self, username):
        target = self.user_repo.getUserByUsername(username)
        if not target:
            return build_response(4062, False, "用户不存在", {})
        if target["role"] == "admin":
            return build_response(4063, False, "管理员账号不可删除", {})
        self.user_repo.deleteUser(username)
        return build_response(0, True, "删除用户成功", {})

    @staticmethod
    def _to_image(image_file):
        if hasattr(image_file, "shape") and hasattr(image_file, "dtype"):
            return image_file
        if cv2 is not None and isinstance(image_file, str) and image_file.strip():
            return cv2.imread(image_file)
        return None


# ========= 界面层 =========
class FaceSystemUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸识别系统")
        self.resize(1100, 700)

        self.service = FaceBusinessService()
        self.current_user = None
        self.current_user_id = None
        self.current_role = None

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.pages = {}
        self.pages["start"] = self.create_start_page()
        self.pages["login"] = self.create_login_page()
        self.pages["register"] = self.create_register_page()
        self.pages["admin_main"] = self.create_admin_main_page()
        self.pages["maintain"] = self.create_maintain_page()
        self.pages["add_user"] = self.create_add_user_page()
        self.pages["edit_user"] = self.create_edit_user_page()
        self.pages["delete_user"] = self.create_delete_user_page()
        self.pages["main"] = self.create_main_page()

        for p in self.pages.values():
            self.stack.addWidget(p)

        self.go("start")

    def go(self, name):
        self.stack.setCurrentWidget(self.pages[name])

    def wrap_center(self, inner_widget, max_width=460):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(30, 25, 30, 25)

        title = QLabel("人脸识别系统")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignHCenter)
        outer.addWidget(title)
        outer.addSpacing(10)

        row = QHBoxLayout()
        row.addStretch()

        card = QFrame()
        card.setObjectName("card")
        card.setMaximumWidth(max_width)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(26, 24, 26, 24)
        card_layout.addWidget(inner_widget)

        row.addWidget(card)
        row.addStretch()

        outer.addLayout(row)
        outer.addStretch()
        return root

    def section_title(self, txt):
        lb = QLabel(txt)
        lb.setObjectName("subtitle")
        lb.setAlignment(Qt.AlignCenter)
        return lb

    def info(self, msg):
        QMessageBox.information(self, "提示", msg)

    def err(self, msg):
        QMessageBox.warning(self, "错误", msg)

    def _refresh_secondary_style(self, root_widget):
        for b in root_widget.findChildren(QPushButton):
            if b.property("class") == "secondary":
                b.setObjectName("secondary_btn")
                b.setStyleSheet(
                    """
                    QPushButton#secondary_btn {
                        background-color: #eef2f9;
                        color: #334155;
                        font-weight: 600;
                    }
                    QPushButton#secondary_btn:hover {
                        background-color: #e3e9f4;
                    }
                    QPushButton#secondary_btn:pressed {
                        background-color: #d8e1f0;
                    }
                """
                )

    def make_panel(self, title_text, placeholder_text=None, content_widget=None):
        frame = QFrame()
        frame.setObjectName("card")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(10)

        t = QLabel(title_text)
        t.setObjectName("panelTitle")
        lay.addWidget(t)

        box = QFrame()
        box.setStyleSheet(
            "QFrame { background: #f8faff; border: 1px dashed #cad5e6; border-radius: 10px; }"
        )
        box_lay = QVBoxLayout(box)

        if content_widget is not None:
            box_lay.addWidget(content_widget)
        else:
            ph = QLabel(placeholder_text or "")
            ph.setObjectName("placeholder")
            ph.setAlignment(Qt.AlignCenter)
            box_lay.addWidget(ph)

        lay.addWidget(box, 1)
        return frame

    def create_start_page(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(18)

        title = QLabel("欢迎使用人脸识别系统")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)

        card = QFrame()
        card.setObjectName("card")
        card.setMaximumWidth(520)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(30, 28, 30, 28)
        card_layout.setSpacing(20)

        btn_row = QHBoxLayout()
        btn_login = QPushButton("登录")
        btn_register = QPushButton("注册")
        btn_login.clicked.connect(lambda: self.go("login"))
        btn_register.clicked.connect(lambda: self.go("register"))
        btn_row.addWidget(btn_login)
        btn_row.addWidget(btn_register)

        card_layout.addWidget(self.section_title("请选择操作"))
        card_layout.addLayout(btn_row)

        layout.addWidget(title)
        layout.addWidget(card, alignment=Qt.AlignCenter)
        return w

    def create_login_page(self):
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setSpacing(14)
        lay.addWidget(self.section_title("登录"))

        self.login_user = QLineEdit()
        self.login_user.setPlaceholderText("用户名")
        self.login_pwd = QLineEdit()
        self.login_pwd.setPlaceholderText("密码")
        self.login_pwd.setEchoMode(QLineEdit.Password)

        lay.addWidget(self.login_user)
        lay.addWidget(self.login_pwd)

        row = QHBoxLayout()
        btn_ok = QPushButton("登录")
        btn_back = QPushButton("返回")
        btn_back.setProperty("class", "secondary")
        btn_ok.clicked.connect(self.do_login)
        btn_back.clicked.connect(lambda: self.go("start"))
        row.addWidget(btn_ok)
        row.addWidget(btn_back)
        lay.addLayout(row)

        page = self.wrap_center(body, 440)
        self._refresh_secondary_style(page)
        return page

    def create_register_page(self):
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setSpacing(14)
        lay.addWidget(self.section_title("注册"))

        self.reg_user = QLineEdit()
        self.reg_user.setPlaceholderText("用户名")
        self.reg_pwd = QLineEdit()
        self.reg_pwd.setPlaceholderText("密码")
        self.reg_pwd.setEchoMode(QLineEdit.Password)
        self.reg_phone = QLineEdit()
        self.reg_phone.setPlaceholderText("手机号")
        self.reg_email = QLineEdit()
        self.reg_email.setPlaceholderText("邮箱")

        lay.addWidget(self.reg_user)
        lay.addWidget(self.reg_pwd)
        lay.addWidget(self.reg_phone)
        lay.addWidget(self.reg_email)

        row = QHBoxLayout()
        btn_ok = QPushButton("注册")
        btn_back = QPushButton("返回")
        btn_back.setProperty("class", "secondary")
        btn_ok.clicked.connect(self.do_register)
        btn_back.clicked.connect(lambda: self.go("start"))
        row.addWidget(btn_ok)
        row.addWidget(btn_back)
        lay.addLayout(row)

        page = self.wrap_center(body, 440)
        self._refresh_secondary_style(page)
        return page

    def create_admin_main_page(self):
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setSpacing(14)
        lay.addWidget(self.section_title("系统管理模式"))

        btn_maintain = QPushButton("用户信息维护")
        btn_logout = QPushButton("退出到起始页")
        btn_logout.setProperty("class", "secondary")
        btn_maintain.clicked.connect(lambda: self.go("maintain"))
        btn_logout.clicked.connect(lambda: self.go("start"))

        lay.addWidget(btn_maintain)
        lay.addWidget(btn_logout)

        page = self.wrap_center(body, 460)
        self._refresh_secondary_style(page)
        return page

    def create_maintain_page(self):
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setSpacing(12)
        lay.addWidget(self.section_title("用户信息维护"))

        btn_add = QPushButton("增加新用户 + 录入人脸")
        btn_edit = QPushButton("修改现有用户")
        btn_del = QPushButton("删除现有用户")
        btn_back = QPushButton("返回")
        btn_back.setProperty("class", "secondary")

        btn_add.clicked.connect(lambda: self.go("add_user"))
        btn_edit.clicked.connect(lambda: self.go("edit_user"))
        btn_del.clicked.connect(lambda: self.go("delete_user"))
        btn_back.clicked.connect(lambda: self.go("admin_main"))

        lay.addWidget(btn_add)
        lay.addWidget(btn_edit)
        lay.addWidget(btn_del)
        lay.addWidget(btn_back)

        page = self.wrap_center(body, 520)
        self._refresh_secondary_style(page)
        return page

    def create_add_user_page(self):
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setSpacing(12)
        lay.addWidget(self.section_title("新增用户并可选录入人脸"))

        self.add_user_name = QLineEdit()
        self.add_user_name.setPlaceholderText("用户名")
        self.add_user_pwd = QLineEdit()
        self.add_user_pwd.setPlaceholderText("密码")
        self.add_user_pwd.setEchoMode(QLineEdit.Password)
        self.add_user_phone = QLineEdit()
        self.add_user_phone.setPlaceholderText("手机号")
        self.add_user_email = QLineEdit()
        self.add_user_email.setPlaceholderText("邮箱")
        self.add_user_photo = QLineEdit()
        self.add_user_photo.setPlaceholderText("人脸图片路径（可选）")

        lay.addWidget(self.add_user_name)
        lay.addWidget(self.add_user_pwd)
        lay.addWidget(self.add_user_phone)
        lay.addWidget(self.add_user_email)
        lay.addWidget(self.add_user_photo)

        row = QHBoxLayout()
        ok = QPushButton("确定")
        back = QPushButton("返回")
        back.setProperty("class", "secondary")
        ok.clicked.connect(self.do_add_user)
        back.clicked.connect(lambda: self.go("maintain"))
        row.addWidget(ok)
        row.addWidget(back)
        lay.addLayout(row)

        page = self.wrap_center(body, 560)
        self._refresh_secondary_style(page)
        return page

    def create_edit_user_page(self):
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setSpacing(12)
        lay.addWidget(self.section_title("用户信息修改"))

        self.edit_old = QLineEdit()
        self.edit_old.setPlaceholderText("原用户名")
        self.edit_new = QLineEdit()
        self.edit_new.setPlaceholderText("新用户名")
        self.edit_pwd = QLineEdit()
        self.edit_pwd.setPlaceholderText("新密码（可留空）")
        self.edit_pwd.setEchoMode(QLineEdit.Password)
        self.edit_photo = QLineEdit()
        self.edit_photo.setPlaceholderText("新照片路径（可选）")

        lay.addWidget(self.edit_old)
        lay.addWidget(self.edit_new)
        lay.addWidget(self.edit_pwd)
        lay.addWidget(self.edit_photo)

        row = QHBoxLayout()
        ok = QPushButton("确定")
        back = QPushButton("返回")
        back.setProperty("class", "secondary")
        ok.clicked.connect(self.do_edit_user)
        back.clicked.connect(lambda: self.go("maintain"))
        row.addWidget(ok)
        row.addWidget(back)
        lay.addLayout(row)

        page = self.wrap_center(body, 520)
        self._refresh_secondary_style(page)
        return page

    def create_delete_user_page(self):
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setSpacing(12)
        lay.addWidget(self.section_title("删除现有用户"))

        self.del_user_name = QLineEdit()
        self.del_user_name.setPlaceholderText("用户名")
        lay.addWidget(self.del_user_name)

        row = QHBoxLayout()
        ok = QPushButton("确认")
        back = QPushButton("取消返回")
        back.setProperty("class", "secondary")
        ok.clicked.connect(self.do_delete_user)
        back.clicked.connect(lambda: self.go("maintain"))
        row.addWidget(ok)
        row.addWidget(back)
        lay.addLayout(row)

        page = self.wrap_center(body, 460)
        self._refresh_secondary_style(page)
        return page

    def create_main_page(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(24, 18, 24, 18)
        outer.setSpacing(14)

        title = QLabel("主界面")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        outer.addWidget(title)

        top = QGridLayout()
        top.setHorizontalSpacing(14)

        self.camera_panel = CameraPanel()
        self.camera_panel.frame_captured.connect(self.on_frame_captured)
        video_panel = self.make_panel("实时视频监控区", content_widget=self.camera_panel)

        result_content = QWidget()
        result_layout = QVBoxLayout(result_content)
        self.result_user = QLabel("用户: -")
        self.result_id = QLabel("用户ID: -")
        self.result_similarity = QLabel("相似度: -")
        self.result_status = QLabel("结果: -")
        btn_recognize = QPushButton("执行识别")
        btn_recognize.clicked.connect(self.do_recognize)
        result_layout.addWidget(self.result_user)
        result_layout.addWidget(self.result_id)
        result_layout.addWidget(self.result_similarity)
        result_layout.addWidget(self.result_status)
        result_layout.addWidget(btn_recognize)
        result_panel = self.make_panel("识别结果反馈区", content_widget=result_content)

        top.addWidget(video_panel, 0, 0)
        top.addWidget(result_panel, 0, 1)

        bottom = QGridLayout()
        bottom.setHorizontalSpacing(14)

        cam_content = QWidget()
        cam_layout = QVBoxLayout(cam_content)
        self.cam_status = QLabel("状态: 待连接")
        self.cam_info = QLabel("设备: camera:0")
        cam_layout.addWidget(self.cam_status)
        cam_layout.addWidget(self.cam_info)
        cam_status_panel = self.make_panel("摄像头状态", content_widget=cam_content)

        match_content = QWidget()
        match_layout = QVBoxLayout(match_content)
        self.match_desc = QLabel("识别阈值: 0.82")
        self.log_hint = QLabel("日志: 自动记录 device_info")
        match_layout.addWidget(self.match_desc)
        match_layout.addWidget(self.log_hint)
        match_info = self.make_panel("匹配结果展示区", content_widget=match_content)

        bottom.addWidget(cam_status_panel, 0, 0)
        bottom.addWidget(match_info, 0, 1)

        outer.addLayout(top, 4)
        outer.addLayout(bottom, 2)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        back = QPushButton("返回起始页（模拟退出）")
        back.setProperty("class", "secondary")
        back.clicked.connect(lambda: self.go("start"))
        btn_row.addWidget(back)
        outer.addLayout(btn_row)

        self._refresh_secondary_style(root)
        return root

    def on_frame_captured(self, _frame):
        self.cam_status.setText("状态: 已连接")

    def do_login(self):
        username = self.login_user.text().strip()
        password = self.login_pwd.text().strip()
        resp = self.service.login(username, password)
        if not resp["success"]:
            self.err(resp["message"])
            return

        self.current_user = resp["data"]["username"]
        self.current_user_id = resp["data"]["user_id"]
        self.current_role = resp["data"]["role"]
        self.info(resp["message"])
        self.go("admin_main" if self.current_role == "admin" else "main")

    def do_register(self):
        username = self.reg_user.text().strip()
        password = self.reg_pwd.text().strip()
        phone = self.reg_phone.text().strip()
        email = self.reg_email.text().strip()

        resp = self.service.register(username, password, phone, email)
        if not resp["success"]:
            self.err(resp["message"])
            return

        self.info(f"{resp['message']}，用户编号: {resp['data']['user_id']}")
        self.go("login")

    def do_add_user(self):
        if self.current_role != "admin":
            self.err("仅管理员可以新增用户")
            return

        username = self.add_user_name.text().strip()
        password = self.add_user_pwd.text().strip()
        phone = self.add_user_phone.text().strip()
        email = self.add_user_email.text().strip()
        photo_path = self.add_user_photo.text().strip()

        reg_resp = self.service.register(username, password, phone, email)
        if not reg_resp["success"]:
            self.err(reg_resp["message"])
            return

        user_id = reg_resp["data"]["user_id"]
        if photo_path:
            face_resp = self.service.addFaceFeature(user_id, photo_path, self.current_user_id)
            if not face_resp["success"]:
                self.err(f"用户已创建，但人脸录入失败: {face_resp['message']}")
                return
            self.info(f"新增成功，feature_id={face_resp['data']['feature_id']}")
            return

        self.info(f"新增用户成功，user_id={user_id}")

    def do_edit_user(self):
        resp = self.service.updateUser(
            self.edit_old.text().strip(),
            self.edit_new.text().strip(),
            self.edit_pwd.text().strip(),
            self.edit_photo.text().strip(),
        )
        if resp["success"]:
            self.info(resp["message"])
        else:
            self.err(resp["message"])

    def do_delete_user(self):
        resp = self.service.deleteUser(self.del_user_name.text().strip())
        if resp["success"]:
            self.info(resp["message"])
        else:
            self.err(resp["message"])

    def do_recognize(self):
        capture_resp = self.camera_panel.captureFrame(camera_id=0, resolution="640x480", frame_rate=25)
        if not capture_resp["success"]:
            self.cam_status.setText("状态: 采集失败")
            self.err(capture_resp["message"])
            return

        self.cam_status.setText("状态: 已连接")
        self.cam_info.setText(
            f"设备: camera:{capture_resp['data']['camera_id']}  分辨率: {capture_resp['data']['resolution']}"
        )

        frame = capture_resp["data"]["frame_data"]
        request_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        recog_resp = self.service.recognizeFace(frame, "camera:0", request_time)

        data = recog_resp["data"]
        self.result_user.setText(f"用户: {data.get('username') or '-'}")
        self.result_id.setText(f"用户ID: {data.get('user_id') or '-'}")
        self.result_similarity.setText(f"相似度: {data.get('similarity', 0.0):.4f}")
        self.result_status.setText(f"结果: {data.get('result', '-')}")

        if recog_resp["success"]:
            self.info(recog_resp["message"])
        else:
            self.err(recog_resp["message"])

    def closeEvent(self, event):
        try:
            if hasattr(self, "camera_panel") and self.camera_panel:
                self.camera_panel.close()
        except Exception:
            pass
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    win = FaceSystemUI()
    win.show()
    sys.exit(app.exec_())
