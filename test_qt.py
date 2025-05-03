import sys
import traceback
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QMessageBox

def main():
    try:
        print("开始测试...")
        app = QApplication(sys.argv)
        window = QMainWindow()
        window.setGeometry(100, 100, 300, 200)
        window.setWindowTitle('测试窗口')
        
        button = QPushButton('测试按钮', window)
        button.move(100, 80)
        
        window.show()
        print("窗口已显示")
        return app.exec_()
    except Exception as e:
        print(f"错误: {e}")
        print(traceback.format_exc())
        QMessageBox.critical(None, "错误", str(e))
        return 1

if __name__ == '__main__':
    print("程序开始")
    sys.exit(main()) 