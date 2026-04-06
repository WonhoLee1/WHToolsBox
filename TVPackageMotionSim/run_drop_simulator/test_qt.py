from PySide6 import QtWidgets
import sys
app = QtWidgets.QApplication(sys.argv)
w = QtWidgets.QMainWindow()
w.setWindowTitle("Test Window")
w.show()
print("GUI Loop Started...")
# 3초 후 자동 종료 (테스트용)
from PySide6 import QtCore
QtCore.QTimer.singleShot(3000, app.quit)
sys.exit(app.exec())
