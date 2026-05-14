import sys
import math
import os
import hashlib
import importlib
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_ROOT = PROJECT_ROOT / "database"
if DATABASE_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, DATABASE_ROOT.as_posix())

from src.db.connection import DbConfig, connect
from src.db.dao import (
    delete_user_by_username,
    get_config_by_key,
    get_user_by_id,
    get_user_by_username,
    get_all_users,
    init_schema,
    insert_face_feature,
    insert_recognition_log,
    insert_user,
    iter_all_active_features,
    update_config,
    update_user_info,
    update_user_status,
)
from src.feature.matcher import match_best


def get_db_conn():
    db_path = DATABASE_ROOT / "data" / "app.sqlite3"
    schema_path = DATABASE_ROOT / "src" / "db" / "schema.sql"
    conn = connect(DbConfig(path=db_path))
    schema_sql = schema_path.read_text(encoding="utf-8")
    init_schema(conn, schema_sql)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(user)")}
    if "photo" not in columns:
        conn.execute("ALTER TABLE user ADD COLUMN photo TEXT")
        conn.commit()
    return conn

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
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
def get_qss(scale=1.0):
    def s(val): return int(val * scale)
    return f"""
    QMainWindow, QWidget {{
        background-color: #f5f7fb;
        font-family: "Microsoft YaHei";
        font-size: {s(15)}px;
    }}
    #titleLabel {{
        font-size: {s(32)}px;
        font-weight: 700;
        color: #1f2d3d;
    }}
    #card {{
        background: white;
        border: 1px solid #e6ebf2;
        border-radius: {s(16)}px;
    }}
    #subtitle {{
        font-size: {s(22)}px;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: {s(10)}px;
    }}
    QLineEdit {{
        min-height: {s(44)}px;
        border: 1px solid #d7deea;
        border-radius: {s(8)}px;
        padding: 0 {s(14)}px;
        background: #fcfdff;
        font-size: {s(15)}px;
    }}
    QLineEdit:focus {{
        border: 1px solid #4c8bf5;
    }}
    QPushButton {{
        min-height: {s(44)}px;
        border: none;
        border-radius: {s(8)}px;
        padding: 0 {s(24)}px;
        background-color: #4c8bf5;
        color: white;
        font-size: {s(16)}px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: #3f7ee8;
    }}
    QPushButton:pressed {{
        background-color: #326fdf;
    }}
    #panelTitle {{
        font-size: {s(16)}px;
        font-weight: 600;
        color: #2c3e50;
    }}
    #placeholder {{
        color: #7f8c9f;
    }}
    """



def build_response(code, success, message, data=None):
    return {
        "code": code,
        "success": success,
        "message": message,
        "data": data or {},
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ========= 算法层（对接仓库内 face_engine）=========
FACE_ENGINE_SRC = PROJECT_ROOT / "face_engine" / "src"
FACE_ENGINE_MODEL_CACHE = PROJECT_ROOT / "face_engine" / ".model_cache"


class FaceAlgorithm:
    """
    使用仓库 `face_engine`：检测+对齐到 160×160，再 FaceNet 提 512 维向量。
    摄像头/OpenCV 帧为 BGR，此处统一转为 RGB 再送入预处理。
    """

    def __init__(self, *, device: str | None = None) -> None:
        self._device = device
        self._preprocess_face_image = None
        self._extract_from_aligned_face = None

    def _ensure_face_engine(self) -> None:
        if self._preprocess_face_image is not None:
            return
        src = FACE_ENGINE_SRC
        if not src.is_dir():
            raise FileNotFoundError(f"未找到 face_engine 源码目录: {src}")
        p = src.as_posix()
        if p not in sys.path:
            sys.path.insert(0, p)
        try:
            from face_engine import extract_from_aligned_face, preprocess_face_image
        except ImportError as e:
            raise ImportError(
                "无法加载 face_engine（通常缺少 torch 等依赖）。"
                "请在 UI 的 venv 中执行: pip install -r ../face_engine/requirements.txt"
            ) from e
        self._preprocess_face_image = preprocess_face_image
        self._extract_from_aligned_face = extract_from_aligned_face

    @staticmethod
    def _to_rgb_numpy(image) -> "np.ndarray":
        if np is None:
            raise RuntimeError("未安装 NumPy")
        if isinstance(image, str) and image.strip():
            if cv2 is None:
                raise RuntimeError("未安装 OpenCV，无法读取图像路径")
            bgr = cv2.imread(image)
            if bgr is None:
                raise RuntimeError(f"无法读取图像: {image}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise RuntimeError(f"图像维度无效: {arr.shape}")
        if arr.dtype not in (np.uint8, np.float32, np.float64):
            arr = arr.astype(np.uint8)
        if arr.dtype != np.uint8 and np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        if cv2 is not None:
            return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr

    def process_to_feature_vector(self, image) -> dict:
        """
        返回 dict:
          - 成功: {success: True, feature_vector: np.ndarray(512,), code: 0}
          - 失败: {success: False, message: str, code: int}
        """
        if image is None:
            return {"success": False, "message": "图像为空", "code": 4041}
        try:
            self._ensure_face_engine()
        except Exception as e:
            return {"success": False, "message": str(e), "code": 5005}

        try:
            rgb = self._to_rgb_numpy(image)
        except Exception as e:
            return {"success": False, "message": str(e), "code": 4033}

        try:
            aligned = self._preprocess_face_image(rgb, device=self._device)
        except RuntimeError as e:
            err = str(e).lower()
            if "no face" in err or "confidence" in err or "below threshold" in err:
                return {"success": False, "message": "未检测到人脸", "code": 4042}
            return {"success": False, "message": str(e), "code": 4042}

        try:
            vec = self._extract_from_aligned_face(
                aligned,
                l2_normalize_output=True,
                device=self._device,
                model_cache=FACE_ENGINE_MODEL_CACHE,
            )
        except Exception as e:
            return {"success": False, "message": str(e), "code": 4043}

        return {"success": True, "feature_vector": vec, "code": 0}


# ========= 业务逻辑层 =========
class FaceBusinessService:
    def __init__(self):
        self.conn = get_db_conn()
        self.algorithm = FaceAlgorithm()
        self._seed_demo_users()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

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

    @staticmethod
    def _row_to_user_dict(row):
        return {
            "user_id": row.user_id,
            "username": row.username,
            "password": row.password,
            "phone": row.phone,
            "email": row.email,
            "status": row.status,
            "role": row.role,
        }

    def _username_exists(self, username):
        return get_user_by_username(self.conn, username) is not None

    def _phone_or_email_exists(self, phone, email):
        if phone:
            phone_row = self.conn.execute("SELECT 1 FROM user WHERE phone = ?", (phone,)).fetchone()
            if phone_row is not None:
                return True
        if email:
            email_row = self.conn.execute("SELECT 1 FROM user WHERE email = ?", (email,)).fetchone()
            if email_row is not None:
                return True
        return False

    def _seed_demo_users(self):
        if self._username_exists("admin"):
            return
        insert_user(
            self.conn,
            username="admin",
            password_hash=self._hash_password("admin123"),
            phone="13800000000",
            email="admin@example.com",
            role="admin",
        )
        insert_user(
            self.conn,
            username="user1",
            password_hash=self._hash_password("123456"),
            phone="13900000000",
            email="user1@example.com",
            role="user",
        )

    def getUserByUsername(self, username):
        user = get_user_by_username(self.conn, username)
        if not user:
            return build_response(4040, False, "用户不存在", {})
        return build_response(0, True, "查询成功", {"user": self._row_to_user_dict(user)})

    def getAllUsers(self):
        users = get_all_users(self.conn)
        return build_response(0, True, "查询成功", {"users": [self._row_to_user_dict(u) for u in users]})

    def login(self, username, password):
        user = get_user_by_username(self.conn, username)
        if not user:
            return build_response(4011, False, "用户未注册", {
                "user_id": None,
                "role": None,
            })

        if int(user.status) != 1:
            return build_response(4013, False, "账号被禁用", {
                "user_id": user.user_id,
                "role": user.role,
            })

        if not self._verify_password(password, user.password):
            return build_response(4012, False, "密码错误", {
                "user_id": user.user_id,
                "role": user.role,
            })

        return build_response(0, True, "登录成功", {
            "user_id": user.user_id,
            "role": user.role,
            "username": user.username,
        })

    def register(self, username, password, phone, email):
        if not username or not password:
            return build_response(4020, False, "用户名和密码不能为空", {"user_id": None})

        if self._username_exists(username) or self._phone_or_email_exists(phone, email):
            return build_response(4021, False, "账号已存在", {"user_id": None})

        password_hash = self._hash_password(password)
        user_id = insert_user(self.conn, username=username, password_hash=password_hash, phone=phone, email=email, role="user")
        return build_response(0, True, "注册成功", {"user_id": user_id})

    def addFaceFeature(self, user_id, image_file, operator_id):
        operator = get_user_by_id(self.conn, operator_id)
        if not operator or operator.role != "admin" or int(operator.status) != 1:
            return build_response(4031, False, "管理员权限不足", {"feature_id": None})

        user = get_user_by_id(self.conn, user_id)
        if not user:
            return build_response(4032, False, "目标用户不存在", {"feature_id": None})

        image = self._to_image(image_file)
        if image is None:
            return build_response(4033, False, "图像格式不合法或路径无效", {"feature_id": None})

        feat = self.algorithm.process_to_feature_vector(image)
        if not feat.get("success"):
            c = int(feat.get("code") or 4035)
            if c == 4042 or "未检测" in str(feat.get("message", "")):
                return build_response(4034, False, "未检测到有效人脸", {"feature_id": None})
            return build_response(4035, False, feat.get("message", "特征提取失败"), {"feature_id": None})

        image_path = image_file if isinstance(image_file, str) else "memory_frame"
        if np is None:
            return build_response(5004, False, "未安装 NumPy，无法录入特征", {"feature_id": None})

        feature_id = insert_face_feature(
            self.conn,
            user_id=int(user_id),
            feature_vector=np.asarray(feat["feature_vector"], dtype=np.float32),
            image_path=image_path,
        )
        return build_response(0, True, "录入成功", {"feature_id": feature_id})

    def recognizeFace(self, image_data, device_info, request_time):
        if image_data is None:
            return build_response(4041, False, "图像为空", {
                "user_id": None,
                "username": None,
                "similarity": 0.0,
                "result": "识别失败",
            })

        feat = self.algorithm.process_to_feature_vector(image_data)
        if not feat.get("success"):
            insert_recognition_log(
                self.conn,
                user_id=None,
                input_image_url="memory_frame",
                similarity=0.0,
                result=0,
                device_info=device_info,
            )
            c = int(feat.get("code") or 4043)
            if c == 4042:
                return build_response(4042, False, "未检测到人脸", {
                    "user_id": None,
                    "username": None,
                    "similarity": 0.0,
                    "result": "未识别",
                })
            return build_response(4043, False, feat.get("message", "特征提取失败"), {
                "user_id": None,
                "username": None,
                "similarity": 0.0,
                "result": "识别失败",
            })

        threshold = float(get_config_by_key(self.conn, "threshold") or get_config_by_key(self.conn, "recognition_threshold") or "0.80")
        feature_vector = np.asarray(feat["feature_vector"], dtype=np.float32) if np is not None else feat["feature_vector"]
        library = list(iter_all_active_features(self.conn))
        compare_resp = match_best(feature_vector, library, threshold=threshold)

        matched = compare_resp.matched
        matched_user_id = compare_resp.matched_user_id
        similarity = float(compare_resp.max_similarity)

        username = None
        result = "陌生人"
        if matched and matched_user_id is not None:
            user = get_user_by_id(self.conn, matched_user_id)
            username = user.username if user else None
            result = "已识别"

        log_id = insert_recognition_log(
            self.conn,
            user_id=matched_user_id if matched else None,
            input_image_url="memory_frame",
            similarity=similarity,
            result=1 if matched else 0,
            device_info=device_info,
        )

        return build_response(0, True, "识别完成", {
            "user_id": matched_user_id if matched else None,
            "username": username,
            "similarity": similarity,
            "result": result,
            "log_id": log_id,
        })

    def updateConfig(self, operator_id, config_key, config_value):
        operator = get_user_by_id(self.conn, operator_id)
        if not operator or operator.role != "admin":
            return build_response(4051, False, "仅管理员可修改配置", {})
        update_config(self.conn, config_key, str(config_value))
        return build_response(0, True, "配置更新成功", {
            "config_key": config_key,
            "config_value": str(config_value),
        })

    def updateUserStatus(self, operator_id, target_username, status):
        operator = get_user_by_id(self.conn, operator_id)
        if not operator or operator.role != "admin":
            return build_response(4052, False, "仅管理员可修改用户状态", {})
        
        target_user = get_user_by_username(self.conn, target_username)
        if not target_user:
            return build_response(4053, False, "目标用户不存在", {})
            
        update_user_status(self.conn, user_id=target_user.user_id, status=status)
        return build_response(0, True, "状态更新成功", {})

    def updateUser(self, operator_id, old_username, new_username, new_password, new_photo):
        if not old_username or not new_username:
            return build_response(4060, False, "用户名不能为空", {})

        old_user = get_user_by_username(self.conn, old_username)
        if not old_user:
            return build_response(4061, False, "原用户名不存在", {})

        if new_username != old_username and get_user_by_username(self.conn, new_username):
            return build_response(4061, False, "新用户名已存在", {})

        password_hash = self._hash_password(new_password) if new_password else None
        update_user_info(
            self.conn,
            user_id=old_user.user_id,
            username=new_username,
            password_hash=password_hash,
            photo=new_photo or None,
        )
        
        # 补充：如果修改了照片，则需要重新提取特征并录入数据库，否则无法识别人脸
        if new_photo:
            face_resp = self.addFaceFeature(old_user.user_id, new_photo, operator_id)
            if not face_resp["success"]:
                return build_response(4064, False, f"用户信息已修改，但人脸特征提取失败: {face_resp['message']}", {})

        return build_response(0, True, "用户信息修改成功", {})

    def deleteUser(self, username):
        target = get_user_by_username(self.conn, username)
        if not target:
            return build_response(4062, False, "用户不存在", {})
        if target.role == "admin":
            return build_response(4063, False, "管理员账号不可删除", {})
        delete_user_by_username(self.conn, username)
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
        self.pages["user_list"] = self.create_user_list_page()
        self.pages["user_detail"] = self.create_user_detail_page()
        self.pages["main"] = self.create_main_page()

        for p in self.pages.values():
            self.stack.addWidget(p)

        self.go("start")

    def go(self, name):
        self.stack.setCurrentWidget(self.pages[name])

    def wrap_center(self, inner_widget, base_width=460):
        # 记录基础宽度以便 resizeEvent 等比缩放
        inner_widget.setProperty("base_width", base_width)
        
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        
        # 内部容器，用于包装 title 和 card
        container = QWidget()
        container.setObjectName("container")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(40, 35, 40, 35)

        title = QLabel("人脸识别系统")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignHCenter)
        container_layout.addWidget(title)
        container_layout.addSpacing(20)

        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(36, 32, 36, 32)
        card_layout.addWidget(inner_widget)

        container_layout.addWidget(card)
        
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(container)
        row.addStretch()

        outer.addStretch()
        outer.addLayout(row)
        outer.addStretch()
        
        # 保存对需要缩放组件的引用
        if not hasattr(self, "_scalable_cards"):
            self._scalable_cards = []
        self._scalable_cards.append(card)
        
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
        w.setProperty("base_width", 520)
        card_layout = QVBoxLayout(w)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(24)

        btn_row = QHBoxLayout()
        btn_login = QPushButton("登录")
        btn_register = QPushButton("注册")
        btn_login.clicked.connect(lambda: self.go("login"))
        btn_register.clicked.connect(lambda: self.go("register"))
        btn_row.addWidget(btn_login)
        btn_row.addWidget(btn_register)

        card_layout.addWidget(self.section_title("请选择操作"))
        card_layout.addLayout(btn_row)

        return self.wrap_center(w, 520)

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
        btn_view_all = QPushButton("查看所有信息")
        btn_logout = QPushButton("退出到起始页")
        btn_logout.setProperty("class", "secondary")
        
        btn_maintain.clicked.connect(lambda: self.go("maintain"))
        btn_view_all.clicked.connect(self.go_to_user_list)
        btn_logout.clicked.connect(lambda: self.go("start"))

        lay.addWidget(btn_maintain)
        lay.addWidget(btn_view_all)
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

    def go_to_user_list(self):
        resp = self.service.getAllUsers()
        if resp.get("success"):
            users = resp["data"].get("users", [])
            self.user_list_widget.clear()
            for u in users:
                item = QListWidgetItem(u["username"])
                item.setData(Qt.UserRole, u)
                self.user_list_widget.addItem(item)
        self.go("user_list")

    def go_to_user_detail(self, item):
        u = item.data(Qt.UserRole)
        self.current_detail_user = u
        self.detail_info_label.setText(
            f"ID: {u.get('user_id', '')}\n"
            f"用户名: {u.get('username', '')}\n"
            f"角色: {u.get('role', '')}\n"
            f"电话: {u.get('phone', '')}\n"
            f"邮箱: {u.get('email', '')}\n"
            f"状态: {u.get('status', '')}"
        )
        self.go("user_detail")

    def create_user_list_page(self):
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setSpacing(12)
        lay.addWidget(self.section_title("所有人员信息"))

        self.user_list_widget = QListWidget()
        self.user_list_widget.itemDoubleClicked.connect(self.go_to_user_detail)
        lay.addWidget(self.user_list_widget)

        tips = QLabel("双击用户名查看详细信息")
        tips.setStyleSheet("color: #7f8c9f; font-size: 12px;")
        tips.setAlignment(Qt.AlignCenter)
        lay.addWidget(tips)

        back = QPushButton("返回")
        back.setProperty("class", "secondary")
        back.clicked.connect(lambda: self.go("admin_main"))
        lay.addWidget(back)

        page = self.wrap_center(body, 520)
        self._refresh_secondary_style(page)
        return page

    def create_user_detail_page(self):
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setSpacing(12)
        lay.addWidget(self.section_title("用户详细信息"))

        self.detail_info_label = QLabel()
        self.detail_info_label.setStyleSheet("font-size: 14px; line-height: 1.5;")
        lay.addWidget(self.detail_info_label)

        row = QHBoxLayout()
        edit_btn = QPushButton("修改数据")
        delete_btn = QPushButton("删除用户")
        back_btn = QPushButton("返回")
        back_btn.setProperty("class", "secondary")
        
        edit_btn.clicked.connect(self.go_to_edit_from_detail)
        delete_btn.clicked.connect(self.do_delete_from_detail)
        back_btn.clicked.connect(lambda: self.go("user_list"))

        row.addWidget(edit_btn)
        row.addWidget(delete_btn)
        row.addWidget(back_btn)
        lay.addLayout(row)

        page = self.wrap_center(body, 520)
        self._refresh_secondary_style(page)
        return page

    def go_to_edit_from_detail(self):
        if hasattr(self, "current_detail_user"):
            self.edit_old.setText(self.current_detail_user["username"])
            self.edit_new.setText("")
            self.edit_pwd.setText("")
            self.edit_photo.setText("")
            self.go("edit_user")

    def do_delete_from_detail(self):
        if hasattr(self, "current_detail_user"):
            username = self.current_detail_user["username"]
            self.del_user_name.setText(username)
            self.do_delete_user()
            self.go_to_user_list()

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
            self.current_user_id,
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
        try:
            if hasattr(self, "service") and self.service:
                self.service.close()
        except Exception:
            pass
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 基准宽度设定为 1100，计算缩放比例
        scale = event.size().width() / 1100.0
        scale = max(0.6, min(scale, 2.5))  # 限制缩放范围在 0.6x 到 2.5x 之间
        
        # 更新全局基础缩放样式
        self.setStyleSheet(get_qss(scale))
        
        # 同步缩放所有的 center card 宽度
        if hasattr(self, "_scalable_cards"):
            for card in self._scalable_cards:
                inner_widget = card.layout().itemAt(0).widget()
                if inner_widget:
                    bw = inner_widget.property("base_width")
                    if bw:
                        new_w = int(bw * scale)
                        card.setFixedWidth(new_w)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceSystemUI()
    window.show()
    sys.exit(app.exec_())
