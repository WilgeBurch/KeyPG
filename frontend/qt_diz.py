import sys
import os
import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QFileIconProvider,
    QLabel, QHBoxLayout, QPushButton, QProgressBar, QSizePolicy, QLineEdit, QDialog, QMessageBox, QInputDialog, QFileDialog, QStyle
)
from PyQt5.QtCore import Qt, QSize, QFileInfo, QEvent, QPoint, QRectF
from PyQt5.QtGui import QPalette, QColor, QFont, QIcon, QPainter, QPixmap, QBrush, QPen, QRegion, QPainterPath

API_URL = "http://localhost:8000"

def set_rounded_mask(widget, radius=13):
    path = QPainterPath()
    rect = QRectF(widget.rect())
    path.addRoundedRect(rect, radius, radius)
    region = QRegion(path.toFillPolygon().toPolygon())
    widget.setMask(region)

class AuthDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.token = None
        self.setWindowTitle("")
        self.setMinimumWidth(350)
        self.setMinimumHeight(220)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.init_ui()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        set_rounded_mask(self, radius=13)

    def init_ui(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#000000"))
        palette.setColor(QPalette.WindowText, QColor("#d4dbe5"))
        palette.setColor(QPalette.Base, QColor("#101012"))
        palette.setColor(QPalette.AlternateBase, QColor("#101012"))
        palette.setColor(QPalette.Text, QColor("#e5eaff"))
        palette.setColor(QPalette.Button, QColor("#18181a"))
        palette.setColor(QPalette.ButtonText, QColor("#e5eaff"))
        palette.setColor(QPalette.Highlight, QColor("#3a4f7a"))
        palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        self.setPalette(palette)

        base_font = QFont("Segoe UI", 11)
        base_font.setLetterSpacing(QFont.AbsoluteSpacing, 0.7)
        self.setFont(base_font)

        self.setStyleSheet("""
            QDialog {
                background-color: #000000;
                color: #d4dbe5;
                font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
                font-size: 11pt;
                border: none;
                border-radius: 13px;
            }
            QLineEdit {
                background: #101012;
                color: #e5eaff;
                border-radius: 4px;
                border: 1px solid #23262b;
                padding: 8px 10px;
                font-size: 11pt;
            }
            QPushButton {
                background-color: #18181a;
                color: #dbe1f1;
                border: 1.2px solid #212125;
                border-radius: 4px;
                padding: 7px 22px;
                font-size: 10.2pt;
                font-weight: 500;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background-color: #292932;
            }
            QPushButton:pressed {
                background-color: #111112;
            }
            QLabel {
                color: #d4dbe5;
            }
        """)

        titlebar = CustomTitleBar(self)
        layout = QVBoxLayout(self)
        layout.setSpacing(13)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(titlebar)
        title = QLabel("Авторизация")
        title.setStyleSheet("font-size: 15pt; font-weight: 500;")
        layout.addWidget(title, alignment=Qt.AlignHCenter)
        self.login_input = QLineEdit()
        self.login_input.setPlaceholderText("Логин")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Пароль")
        self.password_input.setEchoMode(QLineEdit.Password)
        btns = QHBoxLayout()
        self.login_btn = QPushButton("Войти")
        self.register_btn = QPushButton("Зарегистрироваться")
        btns.addWidget(self.login_btn)
        btns.addWidget(self.register_btn)
        layout.addWidget(self.login_input)
        layout.addWidget(self.password_input)
        layout.addLayout(btns)
        self.setLayout(layout)
        self.login_btn.clicked.connect(self.login)
        self.register_btn.clicked.connect(self.register)
        self.login_input.returnPressed.connect(self.focus_password)
        self.password_input.returnPressed.connect(self.login)

    def focus_password(self):
        self.password_input.setFocus()

    def login(self):
        user = self.login_input.text().strip()
        pwd = self.password_input.text().strip()
        if not user or not pwd:
            CustomMessageBox.warning(self, "Ошибка", "Введите логин и пароль")
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
                CustomMessageBox.warning(self, "Ошибка", "Неверный логин или пароль")
        except Exception as e:
            CustomMessageBox.warning(self, "Ошибка", f"Ошибка подключения: {e}")

    def register(self):
        user = self.login_input.text().strip()
        pwd = self.password_input.text().strip()
        if not user or not pwd:
            CustomMessageBox.warning(self, "Ошибка", "Введите логин и пароль")
            return
        data = {
            "username": user,
            "password": pwd
        }
        try:
            resp = requests.post(f"{API_URL}/register", data=data)
            if resp.status_code == 200:
                CustomMessageBox.information(self, "Успех", "Регистрация прошла успешно. Теперь войдите.")
            else:
                CustomMessageBox.warning(self, "Ошибка", f"{resp.json().get('detail', resp.text)}")
        except Exception as e:
            CustomMessageBox.warning(self, "Ошибка", f"Ошибка подключения: {e}")

class CustomTitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setFixedHeight(34)
        self.setStyleSheet("background: #000000; border-top-left-radius:13px; border-top-right-radius:13px;")
        bar = QHBoxLayout(self)
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(0)
        bar.addStretch(1)
        self.btn_min = QPushButton("")
        self.btn_max = QPushButton("")
        self.btn_close = QPushButton("")
        for btn in (self.btn_min, self.btn_max, self.btn_close):
            btn.setFixedSize(40, 34)
            btn.setFlat(True)
            btn.setCursor(Qt.PointingHandCursor)
        self.btn_min.setIcon(self.style().standardIcon(QStyle.SP_TitleBarMinButton))
        self.btn_min.setIconSize(QSize(22, 22))
        self.btn_min.setStyleSheet("""
            QPushButton { background:transparent; }
            QPushButton:hover { background:#19191a; }
        """)
        self.btn_max.setIcon(self.icon_for_maximized(False))
        self.btn_max.setIconSize(QSize(22, 22))
        self.btn_max.setStyleSheet("""
            QPushButton { background:transparent; }
            QPushButton:hover { background:#19191a; }
        """)
        self.btn_close.setIcon(self.style().standardIcon(QStyle.SP_TitleBarCloseButton))
        self.btn_close.setIconSize(QSize(22, 22))
        self.btn_close.setStyleSheet("""
            QPushButton{background:transparent;}
            QPushButton:hover{background:#c0392b;}
        """)
        self.btn_min.clicked.connect(self.on_min)
        self.btn_max.clicked.connect(self.on_max)
        self.btn_close.clicked.connect(self.on_close)
        bar.addWidget(self.btn_min)
        bar.addWidget(self.btn_max)
        bar.addWidget(self.btn_close)
        self.setLayout(bar)
        self.shadow = QWidget(self)
        self.shadow.setFixedHeight(2)
        self.shadow.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #222, stop:1 #19191a);")
        self.shadow.setGeometry(0, self.height()-2, self.width(), 2)

    def icon_for_maximized(self, maximized):
        pix = QPixmap(22, 22)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#eee"))
        pen.setWidth(2)
        painter.setPen(pen)
        if maximized:
            painter.drawRect(4, 8, 10, 8)
            painter.drawRect(8, 4, 10, 8)
        else:
            painter.drawRect(4, 4, 14, 14)
        painter.end()
        return QIcon(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.shadow.setGeometry(0, self.height()-2, self.width(), 2)

    def on_min(self):
        self.parent.showMinimized()

    def on_max(self):
        w = self.parent
        if w.isMaximized():
            w.showNormal()
            self.btn_max.setIcon(self.icon_for_maximized(False))
        else:
            w.showMaximized()
            self.btn_max.setIcon(self.icon_for_maximized(True))

    def on_close(self):
        self.parent.close()

    def mousePressEvent(self, event):
        self.offset = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.parent.move(self.parent.pos() + event.pos() - self.offset)

class CustomMessageBox(QMessageBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet("""
            QMessageBox {
                background-color: #000000;
                color: #d4dbe5;
                font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
                font-size: 11pt;
                border-radius: 13px;
            }
            QLabel {
                color: #d4dbe5;
            }
            QPushButton {
                background-color: #18181a;
                color: #dbe1f1;
                border: 1.2px solid #212125;
                border-radius: 4px;
                padding: 7px 22px;
                font-size: 10.2pt;
                font-weight: 500;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background-color: #292932;
            }
            QPushButton:pressed {
                background-color: #111112;
            }
        """)

    @staticmethod
    def question(parent, title, text, buttons=QMessageBox.Yes|QMessageBox.No, default=QMessageBox.Yes):
        box = CustomMessageBox(parent)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle(title)
        box.setText(text)
        box.setStandardButtons(buttons)
        box.setDefaultButton(default)
        return box.exec_()

    @staticmethod
    def information(parent, title, text, buttons=QMessageBox.Ok, default=QMessageBox.Ok):
        box = CustomMessageBox(parent)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle(title)
        box.setText(text)
        box.setStandardButtons(buttons)
        box.setDefaultButton(default)
        return box.exec_()

    @staticmethod
    def warning(parent, title, text, buttons=QMessageBox.Ok, default=QMessageBox.Ok):
        box = CustomMessageBox(parent)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(title)
        box.setText(text)
        box.setStandardButtons(buttons)
        box.setDefaultButton(default)
        return box.exec_()

class FileDropListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlternatingRowColors(False)
        self.setIconSize(QSize(16, 16))
        self.icon_provider = QFileIconProvider()
        self.viewport().installEventFilter(self)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setStyleSheet("""
            QListWidget::item {
                font-size: 9.5pt;
            }
        """)

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if item is None and event.button() == Qt.LeftButton:
            file_paths, _ = QFileDialog.getOpenFileNames(self, "Выберите файлы для загрузки")
            for file_path in file_paths:
                self.add_file_safe(file_path)
            self.clearSelection()
        else:
            super().mousePressEvent(event)
            if item is None:
                self.clearSelection()

    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseButtonPress:
            pos = event.pos()
            item = self.itemAt(pos)
            if item is None:
                self.clearSelection()
        return super().eventFilter(source, event)

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
            item.setSizeHint(QSize(0, 28))
            self.addItem(item)
        except Exception:
            pass

class FilesListWidget(FileDropListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(False)

class ExitButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(64, 64)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Выйти из аккаунта")
        pix = QPixmap(58, 58)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#f44336"), 6))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(14, 15, 22, 28)
        painter.setBrush(QColor("#f44336"))
        painter.setPen(Qt.NoPen)
        points = [QPoint(44, 29), QPoint(27, 19), QPoint(27, 39)]
        painter.drawPolygon(*points)
        painter.end()
        self.setIcon(QIcon(pix))
        self.setIconSize(QSize(54, 54))
        self.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
            }
            QPushButton:hover {
                background: #18181a;
                border-radius: 10px;
            }
        """)

class FileListWidget(QWidget):
    def __init__(self, token, reauth_callback):
        super().__init__()
        self.token = token
        self.reauth_callback = reauth_callback
        self.setWindowTitle("")
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMinimumSize(550, 500)
        self.init_ui()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        set_rounded_mask(self, radius=13)

    def init_ui(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#000000"))
        palette.setColor(QPalette.WindowText, QColor("#d4dbe5"))
        palette.setColor(QPalette.Base, QColor("#101012"))
        palette.setColor(QPalette.AlternateBase, QColor("#101012"))
        palette.setColor(QPalette.Text, QColor("#e5eaff"))
        palette.setColor(QPalette.Button, QColor("#18181a"))
        palette.setColor(QPalette.ButtonText, QColor("#e5eaff"))
        palette.setColor(QPalette.Highlight, QColor("#3a4f7a"))
        palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        self.setPalette(palette)

        base_font = QFont("Segoe UI", 10)
        base_font.setLetterSpacing(QFont.AbsoluteSpacing, 0.7)
        self.setFont(base_font)

        self.setStyleSheet("""
            QWidget {
                background-color: #000000;
                color: #d4dbe5;
                font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
                font-size: 11pt;
                border-radius: 13px;
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
                background: #101012;
                color: #e5eaff;
                border-radius: 5px;
                border: 1px solid #23262b;
                font-size: 9.5pt;
                padding: 0px 0px 0px 0px;
                outline: none;
            }
            QListWidget::item {
                border-bottom: 1px solid #262a31;
                padding-left: 8px;
                padding-right: 4px;
                padding-top: 4px;
                padding-bottom: 4px;
                margin: 0px;
                min-height: 28px;
                background: transparent;
            }
            QListWidget::item:selected {
                background: #27324a;
                color: #ffffff;
            }
            QPushButton {
                background-color: #18181a;
                color: #dbe1f1;
                border: 1.2px solid #212125;
                border-radius: 4px;
                padding: 7px 22px;
                font-size: 10.2pt;
                font-weight: 500;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background-color: #292932;
            }
            QPushButton:pressed {
                background-color: #111112;
            }
            QProgressBar {
                background: #101012;
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

        titlebar = CustomTitleBar(self)
        main_layout.addWidget(titlebar)

        toprow = QHBoxLayout()
        header = QLabel("Список файлов")
        header.setObjectName("Header")
        toprow.addWidget(header, alignment=Qt.AlignLeft)
        toprow.addStretch(1)
        self.logout_button = ExitButton()
        toprow.addWidget(self.logout_button, alignment=Qt.AlignRight)
        main_layout.addLayout(toprow)

        file_list_label = QLabel("Выбранные файлы для загрузки")
        file_list_label.setObjectName("Section")
        main_layout.addWidget(file_list_label, alignment=Qt.AlignLeft)

        self.file_list = FileDropListWidget()
        self.file_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.file_list, stretch=1)

        upload_row = QHBoxLayout()
        upload_row.setSpacing(10)
        self.upload_button = QPushButton("Загрузить")
        self.upload_button.setMinimumWidth(120)
        upload_row.addWidget(self.upload_button)
        upload_row.addStretch(1)
        main_layout.addLayout(upload_row)

        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(0, 12, 0, 0)
        status_label = QLabel("Список ваших файлов")
        status_label.setObjectName("Section")
        status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        status_layout.addWidget(status_label)

        self.files_list = FilesListWidget()
        self.files_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        status_layout.addWidget(self.files_list)

        bottom_row = QHBoxLayout()
        self.refresh_button = QPushButton("Обновить список")
        self.download_button = QPushButton("Скачать выбранный")
        self.delete_selected_button = QPushButton("Удалить выбранный")
        self.refresh_button.setMinimumWidth(120)
        self.download_button.setMinimumWidth(140)
        self.delete_selected_button.setMinimumWidth(140)
        bottom_row.addWidget(self.refresh_button)
        bottom_row.addWidget(self.download_button)
        bottom_row.addWidget(self.delete_selected_button)
        bottom_row.addStretch(1)
        status_layout.addLayout(bottom_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        main_layout.addLayout(status_layout)

        self.setLayout(main_layout)

        self.upload_button.clicked.connect(self.upload_files)
        self.refresh_button.clicked.connect(self.populate_files)
        self.download_button.clicked.connect(self.download_selected)
        self.delete_selected_button.clicked.connect(self.delete_selected)
        self.logout_button.clicked.connect(self.logout)

        self.populate_files()

    def set_progress_visible(self, visible):
        self.progress_bar.setVisible(visible)
        if not visible:
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("")

    def upload_files(self):
        total = self.file_list.count()
        if not total:
            CustomMessageBox.information(self, "Нет файлов", "Добавьте файлы для загрузки.")
            return

        headers = {'Authorization': f'Bearer {self.token}'}
        self.set_progress_visible(True)
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
                            self.set_progress_visible(False)
                            return
                except Exception as e:
                    self.progress_bar.setFormat(f"Ошибка: {e}")
                    self.set_progress_visible(False)
                    return
        self.progress_bar.setFormat("Все файлы загружены!")
        self.file_list.clear()
        self.populate_files()
        self.set_progress_visible(False)

    def populate_files(self):
        self.files_list.clear()
        headers = {'Authorization': f'Bearer {self.token}'}
        try:
            resp = requests.get(f"{API_URL}/files", headers=headers)
            if resp.status_code == 200:
                for f in resp.json():
                    file_info = QFileInfo(f['filename'] + f['extension'])
                    icon = QFileIconProvider().icon(file_info)
                    item = QListWidgetItem(icon, f"{f['filename']}{f['extension']} ({f['uuid']})")
                    item.setToolTip(f['uuid'])
                    item.setSizeHint(QSize(0, 28))
                    self.files_list.addItem(item)
            else:
                CustomMessageBox.warning(self, "Ошибка", "Не удалось получить список файлов.")
        except Exception as e:
            CustomMessageBox.warning(self, "Ошибка", f"Ошибка при получении списка: {e}")

    def download_selected(self):
        selecteds = self.files_list.selectedItems()
        if not selecteds:
            CustomMessageBox.information(self, "Нет выбора", "Выберите файл для скачивания.")
            return
        for selected in selecteds:
            uuid = selected.toolTip()
            headers = {'Authorization': f'Bearer {self.token}'}
            self.set_progress_visible(True)
            try:
                resp = requests.get(f"{API_URL}/download/{uuid}", headers=headers, stream=True)
                if resp.status_code == 200:
                    file_name, ok = QInputDialog.getText(self, "Сохранить как", "Введите имя файла для сохранения:",
                                                         text=selected.text().split(" (")[0])
                    if not ok or not file_name:
                        self.set_progress_visible(False)
                        continue
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
                    CustomMessageBox.warning(self, "Ошибка", f"Ошибка скачивания: {resp.text}")
            except Exception as e:
                CustomMessageBox.warning(self, "Ошибка", f"Ошибка при скачивании: {e}")
            self.set_progress_visible(False)

    def delete_selected(self):
        selecteds = self.files_list.selectedItems()
        if not selecteds:
            CustomMessageBox.information(self, "Нет выбора", "Выберите файл для удаления.")
            return
        uuids = [item.toolTip() for item in selecteds]
        headers = {'Authorization': f'Bearer {self.token}'}
        confirm = CustomMessageBox.question(self, "Удаление файла", f"Удалить выбранные файлы ({len(uuids)})?",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if confirm == QMessageBox.Yes:
            for uuid in uuids:
                try:
                    resp = requests.delete(f"{API_URL}/delete/{uuid}", headers=headers)
                    if resp.status_code != 0:
                        CustomMessageBox.warning(self, "Ошибка", f"Ошибка удаления: {resp.text}")
                except Exception as e:
                    CustomMessageBox.warning(self, "Ошибка", f"Ошибка при удалении: {e}")
            self.populate_files()

    def logout(self):
        confirm = CustomMessageBox.question(self, "Выход", "Выйти из аккаунта?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if confirm == QMessageBox.Yes:
            self.reauth_callback()

if __name__ == "__main__":
    from functools import partial

    def start_app():
        global window
        auth = AuthDialog()
        if not auth.exec_():
            sys.exit(0)
        window = FileListWidget(token=auth.token, reauth_callback=reauth)
        window.show()

    def reauth():
        window.close()
        start_app()

    app = QApplication(sys.argv)
    start_app()
    sys.exit(app.exec_())