由于之前的版本里面错误上传了venv，请务必重新pull一次确认路径等没有问题
运行本前端模块需要的配置如下：
python=3.12，推荐3.12.9。务必不要使用3.14等实验版，否则会出现编译问题
检查方法：python --version执行后应该输出3.12
如果不是的话请删除原有的venv，执行py -3.12 -m venv venv（替换下面配置流程的第一个，需要自行下载3.12版本的安装包）
配置环境流程如下，请在powershell中运行：
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip（本步骤不是必须
pip install pyqt5 opencv-python bcrypt numpy

