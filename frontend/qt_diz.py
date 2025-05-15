import sys
import os
import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QFileIconProvider,
    QLabel, QHBoxLayout, QPushButton, QProgressBar, QSizePolicy, QLineEdit, QDialog, QMessageBox, QInputDialog
)
from PyQt5.QtCore import Qt, QSize, QFileInfo
from PyQt5.QtGui import QPalette, QColor, QFont

API_URL = "http://localhost:8000"

class AuthDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.token = None
        self.setWindowTitle("Вход или регистрация")
        self.setMinimumWidth(300)
        layout = QVBoxLayout(self)
        self.login_input = QLineEdit()
        self.login_input.setPlaceholderText("Логин")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Пароль")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.login_btn = QPushButton("Войти")
        self.register_btn = QPushButton("Зарегистрироваться")
        layout.addWidget(QLabel("Введите логин и пароль"))
        layout.addWidget(self.login_input)
        layout.addWidget(self.password_input)
        layout.addWidget(self.login_btn)
        layout.addWidget(self.register_btn)
        self.setLayout(layout)
        self.login_btn.clicked.connect(self.login)
        self.register_btn.clicked.connect(self.register)

    def login(self):
        user = self.login_input.text().strip()
        pwd = self.password_input.text().strip()
        if not user or not pwd:
            QMessageBox.warning(self, "Ошибка", "Введите логин и пароль")
            return
        data = {
            "username": user,
            "password": pwd
        }
        try:
            resp = requests.post(f"{API_URL}/token", data=data)
            if resp.status_code == 200:
                self.token = resp.json().get("access_token")
                self.accept()
            else:
                QMessageBox.warning(self, "Ошибка", "Неверный логин или пароль")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка подключения: {e}")

    def register(self):
        user = self.login_input.text().strip()
        pwd = self.password_input.text().strip()
        if not user or not pwd:
            QMessageBox.warning(self, "Ошибка", "Введите логин и пароль")
            return
        data = {
            "username": user,
            "password": pwd
        }
        try:
            resp = requests.post(f"{API_URL}/register", data=data)
            if resp.status_code == 200:
                QMessageBox.information(self, "Успех", "Регистрация прошла успешно. Теперь войдите.")
            else:
                QMessageBox.warning(self, "Ошибка", f"{resp.json().get('detail', resp.text)}")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка подключения: {e}")

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
    def __init__(self, token):
        super().__init__()
        self.token = token
        self.setWindowTitle("Список файлов")
        self.setMinimumSize(550, 450)
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

        file_list_label = QLabel("Выбранные файлы для загрузки")
        file_list_label.setObjectName("Section")
        main_layout.addWidget(file_list_label, alignment=Qt.AlignLeft)

        self.file_list = FileDropListWidget()
        self.file_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.file_list, stretch=1)

        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(10)
        self.upload_button = QPushButton("Загрузить")
        self.refresh_button = QPushButton("Обновить список")
        self.download_button = QPushButton("Скачать выбранный")
        self.upload_button.setMinimumWidth(120)
        self.refresh_button.setMinimumWidth(120)
        self.download_button.setMinimumWidth(120)
        bottom_layout.addWidget(self.upload_button)
        bottom_layout.addWidget(self.refresh_button)
        bottom_layout.addWidget(self.download_button)
        bottom_layout.addStretch(1)
        main_layout.addLayout(bottom_layout)

        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(0, 12, 0, 0)
        status_label = QLabel("Список ваших файлов")
        status_label.setObjectName("Section")
        status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        status_layout.addWidget(status_label)

        self.files_list = QListWidget()
        self.files_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        status_layout.addWidget(self.files_list)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        status_layout.addWidget(self.progress_bar)
        main_layout.addLayout(status_layout)

        self.setLayout(main_layout)

        self.upload_button.clicked.connect(self.upload_files)
        self.refresh_button.clicked.connect(self.populate_files)
        self.download_button.clicked.connect(self.download_selected)

        self.populate_files()

    def upload_files(self):
        total = self.file_list.count()
        if not total:
            QMessageBox.information(self, "Нет файлов", "Добавьте файлы для загрузки.")
            return

        headers = {'Authorization': f'Bearer {self.token}'}
        self.progress_bar.setValue(0)
        for i in range(total):
            file_path = self.file_list.item(i).toolTip()
            if file_path:
                try:
                    with open(file_path, "rb") as f:
                        files = {'file': (os.path.basename(file_path), f)}
                        resp = requests.post(f"{API_URL}/upload", headers=headers, files=files)
                        percent = int(100 * (i+1) / total)
                        self.progress_bar.setValue(percent)
                        if resp.status_code == 200:
                            self.progress_bar.setFormat(f"Загружено {i+1}/{total}")
                        else:
                            self.progress_bar.setFormat(f"Ошибка: {resp.text}")
                            return
                except Exception as e:
                    self.progress_bar.setFormat(f"Ошибка: {e}")
                    return
        self.progress_bar.setFormat("Все файлы загружены!")
        self.file_list.clear()
        self.populate_files()

    def populate_files(self):
        self.files_list.clear()
        headers = {'Authorization': f'Bearer {self.token}'}
        try:
            resp = requests.get(f"{API_URL}/files", headers=headers)
            if resp.status_code == 200:
                for f in resp.json():
                    item = QListWidgetItem(f"{f['filename']}{f['extension']} ({f['uuid']})")
                    item.setToolTip(f['uuid'])
                    self.files_list.addItem(item)
            else:
                QMessageBox.warning(self, "Ошибка", "Не удалось получить список файлов.")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка при получении списка: {e}")

    def download_selected(self):
        selected = self.files_list.currentItem()
        if not selected:
            QMessageBox.information(self, "Нет выбора", "Выберите файл для скачивания.")
            return
        uuid = selected.toolTip()
        headers = {'Authorization': f'Bearer {self.token}'}
        try:
            resp = requests.get(f"{API_URL}/download/{uuid}", headers=headers, stream=True)
            if resp.status_code == 200:
                file_name, ok = QInputDialog.getText(self, "Сохранить как", "Введите имя файла для сохранения:",
                                                     text=selected.text().split(" (")[0])
                if not ok or not file_name:
                    return
                save_path = os.path.join(os.getcwd(), file_name)
                total_length = resp.headers.get('content-length')
                dl = 0
                with open(save_path, "wb") as f:
                    if total_length is None:
                        f.write(resp.content)
                    else:
                        total_length = int(total_length)
                        for chunk in resp.iter_content(chunk_size=4096):
                            if chunk:
                                dl += len(chunk)
                                f.write(chunk)
                                percent = int(100 * dl / total_length)
                                self.progress_bar.setValue(percent)
                self.progress_bar.setFormat("Файл скачан!")
            else:
                QMessageBox.warning(self, "Ошибка", f"Ошибка скачивания: {resp.text}")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка при скачивании: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    auth = AuthDialog()
    if not auth.exec_():
        sys.exit(0)
    window = FileListWidget(token=auth.token)
    window.show()
    sys.exit(app.exec_())
