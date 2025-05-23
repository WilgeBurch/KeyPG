import sys
import os
import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QFileIconProvider,
    QLabel, QHBoxLayout, QPushButton, QProgressBar, QSizePolicy, QLineEdit, QDialog, QMessageBox, QFileDialog,
    QMenu, QAction, QSpacerItem
)
from PyQt5.QtCore import Qt, QSize, QFileInfo, QEvent, QPoint, QRectF
from PyQt5.QtGui import QPalette, QColor, QFont, QIcon, QPainter, QPixmap, QPen, QRegion, QPainterPath

API_URL = "http://localhost:8000"

def human_size(size):
    for unit in ['B','KB','MB','GB','TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"

def set_rounded_mask(widget, radius=13):
    path = QPainterPath()
    rect = QRectF(widget.rect())
    path.addRoundedRect(rect, radius, radius)
    region = QRegion(path.toFillPolygon().toPolygon())
    widget.setMask(region)

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

class MinimalRoundButton(QPushButton):
    # type: "plus", "refresh"
    def __init__(self, icon_type, parent=None):
        super().__init__(parent)
        self.setFixedSize(44, 44)
        self.setCursor(Qt.PointingHandCursor)
        self.icon_type = icon_type
        self.setStyleSheet("""
            QPushButton {
                background: #101012;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: #23262b;
            }
        """)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        size = min(w, h)
        pen = QPen(QColor("#fff"), 3, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)
        if self.icon_type == "plus":
            painter.drawLine(w//2 - size//4, h//2, w//2 + size//4, h//2)
            painter.drawLine(w//2, h//2 - size//4, w//2, h//2 + size//4)
        elif self.icon_type == "refresh":
            # Точная круговая стрелка: дуга и маленький острый уголок (указатель)
            arc_w = size//2
            arc_rect = QRectF((w-arc_w)//2, (h-arc_w)//2, arc_w, arc_w)
            painter.setPen(QPen(QColor("#fff"), 3, Qt.SolidLine, Qt.RoundCap))
            painter.drawArc(arc_rect, 45*16, 270*16)
            # Указатель стрелки
            arrow_length = size // 8
            arrow_width = size // 15
            theta = -45  # угол кончика
            import math
            rx = w//2 + arc_w//2*math.cos(math.radians(45))
            ry = h//2 - arc_w//2*math.sin(math.radians(45))
            painter.save()
            painter.translate(rx, ry)
            painter.rotate(theta)
            painter.drawLine(0, 0, -arrow_length, -arrow_width)
            painter.drawLine(0, 0, -arrow_length, arrow_width)
            painter.restore()
        painter.end()

class CustomProgressBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(44)
        self.bar = QProgressBar(self)
        self.bar.setMinimum(0)
        self.bar.setMaximum(100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.bar.setStyleSheet("""
            QProgressBar {
                background-color: #23262b;
                border-radius: 3px;
                border: none;
                height: 6px;
                margin: 0 0 0 0;
            }
            QProgressBar::chunk {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
                    stop:0 #42e04b, stop:1 #25a525
                );
                border-radius: 3px;
            }
        """)
        self.info_label = QLabel("", self)
        self.bytes_label = QLabel("", self)
        self.info_label.setStyleSheet("color:#e5eaff; font-size: 8.5pt; font-family: 'Segoe UI';")
        self.bytes_label.setStyleSheet("color:#8dfc98; font-size: 8pt; font-family: 'Segoe UI';")
        self.info_label.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        self.bytes_label.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._layout.addWidget(self.info_label)
        self._layout.addWidget(self.bytes_label)
        self._layout.addWidget(self.bar)
        self.setLayout(self._layout)

    def set_progress(self, val, text="", loaded=0, total=0):
        self.bar.setValue(val)
        self.info_label.setText(text)
        if total:
            self.bytes_label.setText(f"{human_size(loaded)} / {human_size(total)}")
        else:
            self.bytes_label.setText("")

    def reset(self):
        self.bar.setValue(0)
        self.info_label.setText("")
        self.bytes_label.setText("")

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
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.context_menu_builder = None

    def set_context_menu_builder(self, builder):
        self.context_menu_builder = builder

    def show_context_menu(self, pos):
        if self.context_menu_builder:
            self.context_menu_builder(self, pos, pos)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            if hasattr(self, "on_delete_via_key") and callable(self.on_delete_via_key):
                self.on_delete_via_key()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        self.last_mouse_pos = event.pos()
        if item is None:
            self.clearSelection()
        else:
            super().mousePressEvent(event)

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

class ExitButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(44, 44)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Выйти из аккаунта")
        pix = QPixmap(36, 36)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#f44336"), 2)
        painter.setPen(pen)
        points = [QPoint(10, 18), QPoint(26, 8), QPoint(26, 28)]
        painter.setBrush(QColor("#f44336"))
        painter.drawPolygon(*points)
        painter.end()
        self.setIcon(QIcon(pix))
        self.setIconSize(QSize(32, 32))
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
        self.setMinimumSize(600, 520)
        self.init_ui()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        shadow_color = QColor(0, 0, 0, 80)
        for i in range(8, 0, -1):
            shadow = rect.adjusted(i, i, -i, -i)
            painter.setPen(Qt.NoPen)
            painter.setBrush(shadow_color)
            painter.drawRoundedRect(shadow, 13, 13)
        painter.setBrush(QColor("#000000"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect.adjusted(4, 4, -4, -4), 13, 13)
        super().paintEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        set_rounded_mask(self, radius=13)

    def build_context_menu_upload(self, widget, pos, global_pos):
        item = widget.itemAt(pos)
        menu = QMenu(widget)
        if item is not None:
            delete_action = QAction("Удалить", widget)
            delete_action.triggered.connect(self.delete_from_upload_list)
            menu.addAction(delete_action)
        else:
            add_action = QAction("Добавить", widget)
            add_action.triggered.connect(self.open_add_files_dialog)
            menu.addAction(add_action)
        menu.exec_(widget.viewport().mapToGlobal(pos))

    def build_context_menu_files(self, widget, pos, global_pos):
        item = widget.itemAt(pos)
        menu = QMenu(widget)
        if item is not None:
            delete_action = QAction("Удалить", widget)
            delete_action.triggered.connect(self.delete_selected)
            menu.addAction(delete_action)
        menu.exec_(widget.viewport().mapToGlobal(pos))

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
                background-color: transparent;
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
        self.file_list.set_context_menu_builder(self.build_context_menu_upload)
        self.file_list.on_delete_via_key = self.delete_from_upload_list
        main_layout.addWidget(self.file_list, stretch=1)

        upload_row = QHBoxLayout()
        upload_row.setSpacing(10)
        self.add_button = MinimalRoundButton("plus")
        upload_row.addWidget(self.add_button)

        upload_bar_col = QVBoxLayout()
        upload_bar_col.setSpacing(0)
        upload_bar_col.setContentsMargins(0, 0, 0, 0)
        self.upload_progress = CustomProgressBar()
        upload_bar_col.addWidget(self.upload_progress)
        upload_row.addLayout(upload_bar_col, stretch=2)
        upload_row.addSpacing(8)
        self.upload_button = QPushButton("Загрузить")
        self.upload_button.setMinimumWidth(120)
        upload_row.addWidget(self.upload_button)
        main_layout.addLayout(upload_row)

        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(0, 12, 0, 0)
        status_label = QLabel("Список ваших файлов")
        status_label.setObjectName("Section")
        status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        status_layout.addWidget(status_label)

        self.files_list = FileDropListWidget()
        self.files_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.files_list.set_context_menu_builder(self.build_context_menu_files)
        self.files_list.on_delete_via_key = self.delete_selected
        status_layout.addWidget(self.files_list)

        bottom_row = QHBoxLayout()
        self.refresh_button = MinimalRoundButton("refresh")
        bottom_row.addWidget(self.refresh_button)

        # --- Изменение: прогрессбар для скачки теперь Expanding между кнопками ---
        self.download_progress = CustomProgressBar()
        self.download_progress.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        bottom_row.addWidget(self.download_progress)

        bottom_row.addSpacing(8)
        # --- Изменение: кнопка теперь "Скачать" ---
        self.download_button = QPushButton("Скачать")
        self.download_button.setMinimumWidth(140)
        bottom_row.addWidget(self.download_button, alignment=Qt.AlignRight)
        status_layout.addLayout(bottom_row)

        main_layout.addLayout(status_layout)
        self.setLayout(main_layout)

        self.upload_button.clicked.connect(self.upload_files)
        self.refresh_button.clicked.connect(self.populate_files)
        self.download_button.clicked.connect(self.download_selected)
        self.logout_button.clicked.connect(self.logout)
        self.add_button.clicked.connect(self.open_add_files_dialog)

        self.populate_files()

    def delete_from_upload_list(self):
        selecteds = self.file_list.selectedItems()
        if not selecteds:
            return
        confirm = CustomMessageBox.question(
            self, "Удаление файла",
            f"Удалить выбранные файлы для загрузки ({len(selecteds)})?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if confirm == QMessageBox.Yes:
            for item in selecteds:
                self.file_list.takeItem(self.file_list.row(item))

    def upload_files(self):
        # --- ИЗМЕНЕНИЕ: теперь загружаются только выбранные файлы, но НЕВЫБРАННЫЕ НЕ УДАЛЯЮТСЯ из списка
        items = self.file_list.selectedItems()
        if not items:
            CustomMessageBox.information(self, "Нет файлов", "Выберите файлы для загрузки.")
            return

        headers = {'Authorization': f'Bearer {self.token}'}
        total_size = sum(os.path.getsize(item.toolTip()) for item in items)
        loaded = 0
        self.upload_progress.set_progress(0, "Загрузка файлов...", 0, total_size)
        self.upload_button.setEnabled(False)
        for idx, item in enumerate(items):
            file_path = item.toolTip()
            try:
                filename = os.path.basename(file_path)
                filesize = os.path.getsize(file_path)
                with open(file_path, "rb") as f:
                    files = {'file': (filename, f)}
                    resp = requests.post(f"{API_URL}/upload", headers=headers, files=files)
                    loaded += filesize
                    percent = int(100 * loaded / total_size) if total_size else 0
                    self.upload_progress.set_progress(
                        percent, f"Загрузка: {filename}", loaded, total_size
                    )
                    if resp.status_code != 200:
                        self.upload_progress.set_progress(0, f"Ошибка загрузки: {resp.text}", loaded, total_size)
                        self.upload_button.setEnabled(True)
                        return
            except Exception as e:
                self.upload_progress.set_progress(0, f"Ошибка: {e}", loaded, total_size)
                self.upload_button.setEnabled(True)
                return
            QApplication.processEvents()
        self.upload_progress.set_progress(100, "Загрузка завершена!", loaded, total_size)
        from threading import Timer
        def reset_bar():
            self.upload_progress.reset()
            self.upload_button.setEnabled(True)
        Timer(0.9, reset_bar).start()
        # Удаляем только загруженные файлы из file_list
        for item in items:
            self.file_list.takeItem(self.file_list.row(item))
        self.populate_files()

    def populate_files(self):
        self.files_list.clear()
        headers = {'Authorization': f'Bearer {self.token}'}
        try:
            resp = requests.get(f"{API_URL}/files", headers=headers)
            if resp.status_code == 200:
                for f in resp.json():
                    file_info = QFileInfo(f['filename'] + f['extension'])
                    icon = QFileIconProvider().icon(file_info)
                    # Добавляем размер в подпись
                    size_str = f"{f['size']} Б" if 'size' in f and f['size'] is not None else "?"
                    item = QListWidgetItem(
                        icon,
                        f"{f['filename']}{f['extension']} ({f['uuid']}, {size_str})"
                    )
                    item.setToolTip(f['uuid'])
                    item.setData(Qt.UserRole, (f['filename'], f['extension']))
                    item.setSizeHint(QSize(0, 28))
                    self.files_list.addItem(item)
            else:
                CustomMessageBox.warning(self, "Ошибка", "Не удалось получить список файлов.")
        except Exception as e:
            CustomMessageBox.warning(self, "Ошибка", f"Ошибка при получении списка: {e}")

    def download_selected(self):
        selecteds = self.files_list.selectedItems()
        if not selecteds:
            CustomMessageBox.information(self, "Нет выбора", "Выберите файл(ы) для скачивания.")
            return
        headers = {'Authorization': f'Bearer {self.token}'}
        # Получаем карту uuid -> size из базы
        files_metadata = {}  # uuid: size
        try:
            resp = requests.get(f"{API_URL}/files", headers=headers)
            if resp.status_code == 200:
                for f in resp.json():
                    files_metadata[f['uuid']] = f.get('size', 0)
        except Exception:
            pass

        files_info = []
        total_size = 0
        for selected in selecteds:
            uuid = selected.toolTip()
            filename, extension = selected.data(Qt.UserRole)
            size = files_metadata.get(uuid)
            # Fallback: если размер не получили — делаем HEAD-запрос
            if size is None or size == 0:
                try:
                    r = requests.head(f"{API_URL}/download/{uuid}", headers=headers)
                    size = int(r.headers.get('content-length', 0))
                except Exception:
                    size = 0
            files_info.append((uuid, filename, extension, size))
            if size > 0:
                total_size += size

        loaded = 0
        self.download_progress.set_progress(0, "Скачивание файлов...", 0, total_size)
        for uuid, filename, extension, filesize in files_info:
            self.download_progress.set_progress(
                int(100 * loaded / total_size) if total_size else 0,
                f"Загрузка: {filename+extension}",
                loaded, total_size
            )
            try:
                resp = requests.get(f"{API_URL}/download/{uuid}", headers=headers, stream=True)
                if resp.status_code == 200:
                    from pathlib import Path
                    downloads_folder = str(Path.home() / "Downloads")
                    save_path = os.path.join(downloads_folder, filename + extension)
                    with open(save_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=4096):
                            if chunk:
                                f.write(chunk)
                                loaded += len(chunk)
                                # Если размер файла был неизвестен, корректируем total_size на лету
                                if filesize == 0:
                                    total_size += len(chunk)
                                percent = int(100 * loaded / total_size) if total_size else 0
                                self.download_progress.set_progress(
                                    percent,
                                    f"Загрузка: {filename+extension}",
                                    loaded,
                                    total_size
                                )
                                QApplication.processEvents()
                    self.download_progress.set_progress(
                        int(100 * loaded / total_size) if total_size else 100,
                        f"Скачан {filename+extension}",
                        loaded,
                        total_size
                    )
                else:
                    self.download_progress.set_progress(0, f"Ошибка скачивания: {resp.text}", loaded, total_size)
            except Exception as e:
                self.download_progress.set_progress(0, f"Ошибка при скачивании: {e}", loaded, total_size)
        from threading import Timer
        def reset_bar():
            self.download_progress.reset()
        Timer(1.2, reset_bar).start()

    def delete_selected(self):
        selecteds = self.files_list.selectedItems()
        if not selecteds:
            CustomMessageBox.information(self, "Нет выбора", "Выберите файл для удаления.")
            return
        uuids = [item.toolTip() for item in selecteds]
        headers = {'Authorization': f'Bearer {self.token}'}
        confirm = CustomMessageBox.question(
            self,
            "Удаление файла",
            f"Удалить выбранные файлы ({len(uuids)})?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if confirm == QMessageBox.Yes:
            for uuid in uuids:
                try:
                    requests.delete(f"{API_URL}/delete/{uuid}", headers=headers)
                except Exception:
                    pass
            self.populate_files()

    def logout(self):
        confirm = CustomMessageBox.question(self, "Выход", "Выйти из аккаунта?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if confirm == QMessageBox.Yes:
            self.reauth_callback()

    def open_add_files_dialog(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Выберите файлы для загрузки")
        for file_path in file_paths:
            self.file_list.add_file_safe(file_path)

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

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        shadow_color = QColor(0, 0, 0, 80)
        for i in range(8, 0, -1):
            shadow = rect.adjusted(i, i, -i, -i)
            painter.setPen(Qt.NoPen)
            painter.setBrush(shadow_color)
            painter.drawRoundedRect(shadow, 13, 13)
        painter.setBrush(QColor("#000000"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect.adjusted(4, 4, -4, -4), 13, 13)
        super().paintEvent(event)

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
                background-color: transparent;
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
        layout.setContentsMargins(32, 32, 32, 32)
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
        self.login_input.installEventFilter(self)
        self.login_input.setFocus()

    def eventFilter(self, obj, event):
        if obj == self.login_input and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.focus_password()
                return True
        return super().eventFilter(obj, event)

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
                # --- ИСПРАВЛЕНИЕ: только одно окно при ошибке (убираем дублирование)
                if resp.status_code == 401 or resp.status_code == 400:
                    CustomMessageBox.warning(self, "Ошибка", "Неверный логин или пароль")
                else:
                    CustomMessageBox.warning(self, "Ошибка", f"Ошибка: {resp.text}")
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
        self.btn_min.setIcon(self._white_min_icon())
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
        self.btn_close.setIcon(self._white_close_icon())
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

    def _white_min_icon(self):
        pix = QPixmap(22, 22)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        pen = QPen(QColor("#fff"), 3)
        painter.setPen(pen)
        painter.drawLine(6, 15, 16, 15)
        painter.end()
        return QIcon(pix)

    def _white_close_icon(self):
        pix = QPixmap(22, 22)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        pen = QPen(QColor("#fff"), 2)
        painter.setPen(pen)
        painter.drawLine(6, 6, 16, 16)
        painter.drawLine(16, 6, 6, 16)
        painter.end()
        return QIcon(pix)

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

if __name__ == "__main__":
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