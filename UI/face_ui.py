import tkinter as tk
from tkinter import messagebox
from camera_widget import CameraPanel


class FaceSystemApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("人脸识别系统")
        self.geometry("980x640")
        self.resizable(False, False)

        # 模拟用户数据（演示用）
        # 真实项目应替换为数据库
        self.users = {
            "admin": {"password": "admin123", "photo": "admin.jpg", "role": "admin"},
            "user1": {"password": "123456", "photo": "user1.jpg", "role": "user"},
        }
        self.current_user = None

        container = tk.Frame(self)
        container.pack(fill="both", expand=True)

        self.frames = {}
        pages = [
            StartPage, LoginPage, RegisterPage,
            AdminMainPage, UserMaintenancePage,
            AddUserPage, EditUserPage, DeleteUserPage,
            MainPage
        ]

        for P in pages:
            page_name = P.__name__
            frame = P(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

    def login(self, username, password):
        if username in self.users and self.users[username]["password"] == password:
            self.current_user = username
            role = self.users[username]["role"]
            if role == "admin":
                self.show_frame("AdminMainPage")
            else:
                self.show_frame("MainPage")
            return True
        return False

    def register_user(self, username, password):
        if not username or not password:
            return False, "用户名和密码不能为空"
        if username in self.users:
            return False, "用户名已存在"
        self.users[username] = {"password": password, "photo": "", "role": "user"}
        return True, "注册成功"

    def add_user(self, username, password, photo):
        if username in self.users:
            return False, "用户已存在"
        self.users[username] = {"password": password, "photo": photo, "role": "user"}
        return True, "新增用户成功"

    def edit_user(self, old_username, new_username, new_password, new_photo):
        if old_username not in self.users:
            return False, "原用户名不存在"
        if new_username != old_username and new_username in self.users:
            return False, "新用户名已存在"

        role = self.users[old_username]["role"]
        self.users.pop(old_username)
        self.users[new_username] = {
            "password": new_password if new_password else "123456",
            "photo": new_photo,
            "role": role
        }
        return True, "用户信息修改成功"

    def delete_user(self, username):
        if username not in self.users:
            return False, "用户不存在"
        if username == "admin":
            return False, "管理员账号不可删除"
        self.users.pop(username)
        return True, "删除用户成功"


class BasePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#f3f3f3")
        self.controller = controller

    def title_label(self, text):
        tk.Label(self, text=text, font=("Microsoft YaHei", 22, "bold"), bg="#f3f3f3").pack(pady=20)


class StartPage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.title_label("人脸识别系统")

        btn_frame = tk.Frame(self, bg="#f3f3f3")
        btn_frame.pack(pady=60)

        tk.Button(btn_frame, text="登录", width=14, height=2,
                  command=lambda: controller.show_frame("LoginPage")).grid(row=0, column=0, padx=30)
        tk.Button(btn_frame, text="注册", width=14, height=2,
                  command=lambda: controller.show_frame("RegisterPage")).grid(row=0, column=1, padx=30)


class LoginPage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.title_label("登录")

        form = tk.Frame(self, bg="#f3f3f3")
        form.pack(pady=20)

        tk.Label(form, text="用户名：", bg="#f3f3f3").grid(row=0, column=0, pady=10)
        self.username_entry = tk.Entry(form, width=28)
        self.username_entry.grid(row=0, column=1)

        tk.Label(form, text="密码：", bg="#f3f3f3").grid(row=1, column=0, pady=10)
        self.password_entry = tk.Entry(form, show="*", width=28)
        self.password_entry.grid(row=1, column=1)

        btns = tk.Frame(self, bg="#f3f3f3")
        btns.pack(pady=20)

        tk.Button(btns, text="登录", width=12, command=self.do_login).grid(row=0, column=0, padx=10)
        tk.Button(btns, text="返回", width=12,
                  command=lambda: controller.show_frame("StartPage")).grid(row=0, column=1, padx=10)

    def do_login(self):
        u = self.username_entry.get().strip()
        p = self.password_entry.get().strip()
        if self.controller.login(u, p):
            messagebox.showinfo("提示", "登录成功")
        else:
            messagebox.showerror("错误", "用户名或密码错误")


class RegisterPage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.title_label("注册")

        form = tk.Frame(self, bg="#f3f3f3")
        form.pack(pady=20)

        tk.Label(form, text="用户名：", bg="#f3f3f3").grid(row=0, column=0, pady=10)
        self.username_entry = tk.Entry(form, width=28)
        self.username_entry.grid(row=0, column=1)

        tk.Label(form, text="密码：", bg="#f3f3f3").grid(row=1, column=0, pady=10)
        self.password_entry = tk.Entry(form, show="*", width=28)
        self.password_entry.grid(row=1, column=1)

        btns = tk.Frame(self, bg="#f3f3f3")
        btns.pack(pady=20)

        tk.Button(btns, text="注册", width=12, command=self.do_register).grid(row=0, column=0, padx=10)
        tk.Button(btns, text="返回", width=12,
                  command=lambda: controller.show_frame("StartPage")).grid(row=0, column=1, padx=10)

    def do_register(self):
        u = self.username_entry.get().strip()
        p = self.password_entry.get().strip()
        ok, msg = self.controller.register_user(u, p)
        if ok:
            messagebox.showinfo("提示", msg)
            self.controller.show_frame("LoginPage")
        else:
            messagebox.showerror("错误", msg)


class AdminMainPage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.title_label("系统管理模式")

        tk.Button(self, text="用户信息维护", width=22, height=2,
                  command=lambda: controller.show_frame("UserMaintenancePage")).pack(pady=10)
        tk.Button(self, text="退出到起始页", width=22, height=2,
                  command=lambda: controller.show_frame("StartPage")).pack(pady=10)


class UserMaintenancePage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.title_label("用户信息维护")

        tk.Button(self, text="增加新用户", width=22, height=2,
                  command=lambda: controller.show_frame("AddUserPage")).pack(pady=8)
        tk.Button(self, text="修改现有用户", width=22, height=2,
                  command=lambda: controller.show_frame("EditUserPage")).pack(pady=8)
        tk.Button(self, text="删除现有用户", width=22, height=2,
                  command=lambda: controller.show_frame("DeleteUserPage")).pack(pady=8)
        tk.Button(self, text="返回", width=22, height=2,
                  command=lambda: controller.show_frame("AdminMainPage")).pack(pady=8)


class AddUserPage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.title_label("新增用户")

        form = tk.Frame(self, bg="#f3f3f3")
        form.pack(pady=20)

        tk.Label(form, text="用户名：", bg="#f3f3f3").grid(row=0, column=0, pady=8)
        self.username = tk.Entry(form, width=28)
        self.username.grid(row=0, column=1)

        tk.Label(form, text="密码：", bg="#f3f3f3").grid(row=1, column=0, pady=8)
        self.password = tk.Entry(form, width=28)
        self.password.grid(row=1, column=1)

        tk.Label(form, text="照片路径：", bg="#f3f3f3").grid(row=2, column=0, pady=8)
        self.photo = tk.Entry(form, width=28)
        self.photo.grid(row=2, column=1)

        tk.Button(self, text="确定", width=12, command=self.submit).pack(pady=10)
        tk.Button(self, text="返回", width=12,
                  command=lambda: controller.show_frame("UserMaintenancePage")).pack()

    def submit(self):
        ok, msg = self.controller.add_user(
            self.username.get().strip(),
            self.password.get().strip(),
            self.photo.get().strip()
        )
        if ok:
            messagebox.showinfo("提示", msg)
        else:
            messagebox.showerror("错误", msg)


class EditUserPage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.title_label("用户信息修改")

        form = tk.Frame(self, bg="#f3f3f3")
        form.pack(pady=20)

        tk.Label(form, text="原用户名：", bg="#f3f3f3").grid(row=0, column=0, pady=8)
        self.old_user = tk.Entry(form, width=28)
        self.old_user.grid(row=0, column=1)

        tk.Label(form, text="新用户名：", bg="#f3f3f3").grid(row=1, column=0, pady=8)
        self.new_user = tk.Entry(form, width=28)
        self.new_user.grid(row=1, column=1)

        tk.Label(form, text="新密码：", bg="#f3f3f3").grid(row=2, column=0, pady=8)
        self.new_pass = tk.Entry(form, width=28)
        self.new_pass.grid(row=2, column=1)

        tk.Label(form, text="新照片路径：", bg="#f3f3f3").grid(row=3, column=0, pady=8)
        self.new_photo = tk.Entry(form, width=28)
        self.new_photo.grid(row=3, column=1)

        btns = tk.Frame(self, bg="#f3f3f3")
        btns.pack(pady=10)
        tk.Button(btns, text="确定", width=12, command=self.submit).grid(row=0, column=0, padx=8)
        tk.Button(btns, text="返回", width=12,
                  command=lambda: controller.show_frame("UserMaintenancePage")).grid(row=0, column=1, padx=8)

    def submit(self):
        ok, msg = self.controller.edit_user(
            self.old_user.get().strip(),
            self.new_user.get().strip(),
            self.new_pass.get().strip(),
            self.new_photo.get().strip()
        )
        if ok:
            messagebox.showinfo("提示", msg)
        else:
            messagebox.showerror("错误", msg)


class DeleteUserPage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.title_label("删除现有用户")

        form = tk.Frame(self, bg="#f3f3f3")
        form.pack(pady=20)

        tk.Label(form, text="用户名：", bg="#f3f3f3").grid(row=0, column=0, pady=8)
        self.username = tk.Entry(form, width=28)
        self.username.grid(row=0, column=1)

        btns = tk.Frame(self, bg="#f3f3f3")
        btns.pack(pady=10)
        tk.Button(btns, text="确认", width=12, command=self.submit).grid(row=0, column=0, padx=8)
        tk.Button(btns, text="取消返回", width=12,
                  command=lambda: controller.show_frame("UserMaintenancePage")).grid(row=0, column=1, padx=8)

    def submit(self):
        ok, msg = self.controller.delete_user(self.username.get().strip())
        if ok:
            messagebox.showinfo("提示", msg)
        else:
            messagebox.showerror("错误", msg)


class MainPage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.title_label("主界面")

        top = tk.Frame(self, bg="#f3f3f3")
        top.pack(pady=10)

        left_cam = tk.LabelFrame(top, text="实时视频监控区", width=420, height=260, bg="white")
        left_cam.grid(row=0, column=0, padx=15)
        left_cam.pack_propagate(False)
        tk.Label(left_cam, text="（摄像头画面占位）", bg="white").pack(expand=True)

        right_result = tk.LabelFrame(top, text="识别结果反馈区", width=420, height=260, bg="white")
        right_result.grid(row=0, column=1, padx=15)
        right_result.pack_propagate(False)
        tk.Label(right_result, text="（目标截图 + 粗略匹配信息）", bg="white").pack(expand=True)

        bottom = tk.Frame(self, bg="#f3f3f3")
        bottom.pack(pady=20)

        cam_status = tk.LabelFrame(bottom, text="摄像头状态", width=420, height=120, bg="white")
        cam_status.grid(row=0, column=0, padx=15)
        cam_status.pack_propagate(False)
        tk.Label(cam_status, text="已连接 / 帧率 / 分辨率", bg="white").pack(expand=True)

        match_info = tk.LabelFrame(bottom, text="匹配结果展示区", width=420, height=120, bg="white")
        match_info.grid(row=0, column=1, padx=15)
        match_info.pack_propagate(False)
        tk.Label(match_info, text="姓名、ID、匹配度、判定状态（是否认识）", bg="white").pack(expand=True)

        tk.Button(self, text="返回起始页（模拟退出）", width=22,
                  command=lambda: controller.show_frame("StartPage")).pack(pady=15)


if __name__ == "__main__":
    app = FaceSystemApp()
    app.mainloop()