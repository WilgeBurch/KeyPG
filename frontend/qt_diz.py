import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QFileIconProvider,
    QLabel, QHBoxLayout, QPushButton, QProgressBar, QSizePolicy
)
from PyQt5.QtCore import Qt, QSize, QFileInfo
from PyQt5.QtGui import QPalette, QColor, QFont

class FileDropListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlternatingRowColors(False)
        self.setIconSize(QSize(20, 20))
        self.icon_provider = QFileIconProvider()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                self.add_file_safe(file_path)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def add_file_safe(self, file_path):
        try:
            if not file_path or not os.path.exists(file_path) or not os.path.isfile(file_path):
                return
            for i in range(self.count()):
                if self.item(i).toolTip() == file_path:
                    return
            filename = os.path.basename(file_path)
            file_info = QFileInfo(file_path)
            icon = self.icon_provider.icon(file_info)
            if icon.isNull():
                icon = self.icon_provider.icon(QFileIconProvider.File)
            item = QListWidgetItem(icon, filename)
            item.setToolTip(file_path)
            item.setSizeHint(QSize(0, 32))
            self.addItem(item)
        except Exception:
            pass

class FileListWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Список файлов")
        self.setMinimumSize(550, 390)
        self.init_ui()

    def init_ui(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#181b20"))
        palette.setColor(QPalette.WindowText, QColor("#d4dbe5"))
        palette.setColor(QPalette.Base, QColor("#23262b"))
        palette.setColor(QPalette.AlternateBase, QColor("#23262b"))
        palette.setColor(QPalette.Text, QColor("#e5eaff"))
        palette.setColor(QPalette.Button, QColor("#21242a"))
        palette.setColor(QPalette.ButtonText, QColor("#e5eaff"))
        palette.setColor(QPalette.Highlight, QColor("#444b5c"))
        palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        self.setPalette(palette)

        base_font = QFont("Segoe UI", 10)
        base_font.setLetterSpacing(QFont.AbsoluteSpacing, 0.7)
        self.setFont(base_font)

        self.setStyleSheet("""
            QWidget {
                background-color: #181b20;
                color: #d4dbe5;
                font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
                font-size: 11pt;
            }
            QLabel#Header {
                font-size: 15pt;
                font-weight: 500;
                padding-bottom: 2px;
                letter-spacing: 0.7px;
                background: none;
                border: none;
            }
            QLabel#Section {
                font-size: 10pt;
                color: #78829a;
                padding-bottom: 1px;
                padding-top: 10px;
                background: none;
                border: none;
            }
            QListWidget {
                background: #22252a;
                color: #e5eaff;
                border-radius: 5px;
                border: 1px solid #23262b;
                font-size: 10.5pt;
                padding: 0px 0px 0px 0px;
                outline: none;
            }
            QListWidget::item {
                border-bottom: 1px solid #262a31;
                padding-left: 8px;
                padding-right: 4px;
                padding-top: 6px;
                padding-bottom: 6px;
                margin: 0px;
                min-height: 32px;
                background: transparent;
            }
            QListWidget::item:selected {
                background: #27324a;
                color: #ffffff;
            }
            QPushButton {
                background-color: #21242a;
                color: #dbe1f1;
                border: 1.2px solid #30343b;
                border-radius: 4px;
                padding: 7px 22px;
                font-size: 10.2pt;
                font-weight: 500;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background-color: #323640;
            }
            QPushButton:pressed {
                background-color: #262a31;
            }
            QProgressBar {
                background: #22252a;
                border-radius: 4px;
                border: 1px solid #292c33;
                height: 16px;
                text-align: center;
                font-size: 10pt;
                margin: 0px 0px 3px 0px;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3d4253, stop:1 #23262b
                );
                border-radius: 4px;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 12)
        main_layout.setSpacing(8)

        header = QLabel("Список файлов")
        header.setObjectName("Header")
        main_layout.addWidget(header, alignment=Qt.AlignLeft)

        file_list_label = QLabel("Выбранные файлы")
        file_list_label.setObjectName("Section")
        main_layout.addWidget(file_list_label, alignment=Qt.AlignLeft)

        self.file_list = FileDropListWidget()
        self.file_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.file_list, stretch=1)

        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(10)
        self.upload_button = QPushButton("Загрузить в базу")
        self.restore_button = QPushButton("Восстановить")
        self.upload_button.setMinimumWidth(120)
        self.restore_button.setMinimumWidth(120)
        bottom_layout.addWidget(self.upload_button)
        bottom_layout.addWidget(self.restore_button)
        bottom_layout.addStretch(1)
        main_layout.addLayout(bottom_layout)

        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(0, 12, 0, 0)
        status_label = QLabel("Статус операции")
        status_label.setObjectName("Section")
        status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        status_layout.addWidget(status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        status_layout.addWidget(self.progress_bar)
        main_layout.addLayout(status_layout)

        self.setLayout(main_layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FileListWidget()
    window.show()
    sys.exit(app.exec_())