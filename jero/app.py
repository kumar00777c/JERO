from PySide6.QtWidgets import QApplication

from jero.ui.window import MainWindow


class JeroApp:
    def __init__(self, argv):
        self.app = QApplication(argv)
        self.window = MainWindow()

    def run(self):
        self.window.show()
        return self.app.exec()