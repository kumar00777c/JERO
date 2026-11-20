import markdown
import re

from pathlib import Path

from PySide6.QtCore import (
    QEasingCurve,
    QParallelAnimationGroup,
    QPropertyAnimation,
    QThread,
    QTimer,
    Qt,
)
from PySide6.QtGui import QIcon,QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QInputDialog,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from jero.brain.local_qwen import LocalQwenBrain
from jero.brain.worker import GenerationWorker
from jero.memory.commands import MemoryCommandHandler


def render_markdown(text):
    # Split inline asterisk bullets into separate Markdown list items.
    text = re.sub(
        r"(?<!\*)\s+\*\s+(?=[A-Z])",
        "\n- ",
        text,
    )

    return markdown.markdown(
        text,
        extensions=["fenced_code", "nl2br"],
    )


class MessageInput(QPlainTextEdit):
    MIN_HEIGHT = 45   # same as the +, mic and arrow buttons
    MAX_HEIGHT = 150

    def __init__(self, send_callback):
        super().__init__()

        self.send_callback = send_callback

        self.setPlaceholderText("Talk to JERO...")

        self.document().setDocumentMargin(0)

        self.setMinimumHeight(self.MIN_HEIGHT)
        self.setMaximumHeight(self.MAX_HEIGHT)
        self.setFixedHeight(self.MIN_HEIGHT)

        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)

        self.textChanged.connect(self.adjust_height)
        self.setStyleSheet("""
            QPlainTextEdit {
                background-color: #222222;
                color: #E6E6E6;
                border: 1px solid #3A3A3A;
                border-radius: 12px;
                padding: 10px 14px;
                font-size: 15px;
                selection-background-color: #3B5B7A;
                selection-color: #FFFFFF;
            }

            QPlainTextEdit:focus {
                border: 1px solid #505050;
            }

            QScrollBar:vertical {
                background: transparent;
                width: 8px;
                margin: 4px;
            }

            QScrollBar::handle:vertical {
                background: #505050;
                border-radius: 4px;
                min-height: 25px;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                self.insertPlainText("\n")
                return

            self.send_callback()
            return

        super().keyPressEvent(event)

    def adjust_height(self):
        if not self.toPlainText():
            self.setFixedHeight(self.MIN_HEIGHT)
            return

        lines = max(1, int(self.document().size().height()))
        line_height = self.fontMetrics().lineSpacing()

        # 10px top + 10px bottom padding + 1px border top and bottom
        new_height = lines * line_height + 20 + 2

        new_height = max(self.MIN_HEIGHT, min(new_height, self.MAX_HEIGHT))
        self.setFixedHeight(new_height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjust_height()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("JERO i1")
        self.setWindowIcon(
            QIcon(str(Path(__file__).resolve().parent / "assets" / "JERO_Icon.ico"))
        )
        self.resize(1100, 700)

        self.brain = LocalQwenBrain()
        self.memory_handler = MemoryCommandHandler(self.brain.memory)

        self.current_response = ""
        self.chat_history = []
        self.pending_attachments = []
        self.current_conversation_id = self.brain.memory.create_conversation()

        # ==================================================
        # MAIN CONTAINER
        # ==================================================

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ==================================================
        # SIDEBAR
        # ==================================================

        self.sidebar = QFrame()
        self.sidebar.setMinimumWidth(250)
        self.sidebar.setMaximumWidth(250)

        # The sidebar slides between 0 and 250 px, like a collapsible
        # ChatGPT-style side panel.
        self.sidebar_animation = QParallelAnimationGroup(self)

        self.sidebar_min_animation = QPropertyAnimation(
            self.sidebar,
            b"minimumWidth",
            self,
        )
        self.sidebar_max_animation = QPropertyAnimation(
            self.sidebar,
            b"maximumWidth",
            self,
        )

        for animation in (
            self.sidebar_min_animation,
            self.sidebar_max_animation,
        ):
            animation.setDuration(240)
            animation.setEasingCurve(QEasingCurve.OutCubic)
            self.sidebar_animation.addAnimation(animation)

        self.sidebar_open = True

        self.sidebar.setStyleSheet("""
            QFrame {
                background-color: #111111;
            }

            QLabel {
                color: #ffffff;
            }

            QPushButton {
                background-color: transparent;
                color: #dddddd;
                border: none;
                border-radius: 8px;
                padding: 10px;
                text-align: left;
                font-size: 14px;
            }

            QPushButton:hover {
                background-color: #222222;
            }

            QPushButton:pressed {
                background-color: #2a2a2a;
            }
        """)

        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(20, 20, 20, 20)
        sidebar_layout.setSpacing(15)


        # JERO logo
        logo = QLabel()

        logo_path = (
            Path(__file__).resolve().parent
            / "assets"
            / "JERO.png"
        )

        pixmap = QPixmap(str(logo_path))

        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                220,
                220,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

            logo.setPixmap(pixmap)

        logo.setAlignment(Qt.AlignCenter)
        logo.setFixedHeight(120)
        logo.setContentsMargins(0, 45, 0, 0)

        sidebar_layout.addWidget(logo)

        # New chat button
        new_chat_button = QPushButton("+  New Chat")
        sidebar_layout.addWidget(new_chat_button)

        # Recent chats
        recent_label = QLabel("Recent Chats")
        recent_label.setStyleSheet("""
            color: #888888;
            font-size: 12px;
            font-weight: bold;
        """)

        sidebar_layout.addWidget(recent_label)

        self.recent_chats_scroll = QScrollArea()
        self.recent_chats_scroll.setWidgetResizable(True)
        self.recent_chats_scroll.setFrameShape(QFrame.NoFrame)
        self.recent_chats_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.recent_chats_scroll.setStyleSheet("background: transparent; border: none;")

        self.recent_chats_widget = QWidget()
        self.recent_chats_layout = QVBoxLayout(self.recent_chats_widget)
        self.recent_chats_layout.setContentsMargins(0, 0, 0, 0)
        self.recent_chats_layout.setSpacing(4)
        self.recent_chats_layout.addStretch()
        self.recent_chats_scroll.setWidget(self.recent_chats_widget)
        sidebar_layout.addWidget(self.recent_chats_scroll, 1)

        new_chat_button.clicked.connect(self.new_chat)
        self.refresh_recent_chats()

        sidebar_layout.addStretch()

        # Settings
        settings_button = QPushButton("⚙  Settings")
        sidebar_layout.addWidget(settings_button)

        main_layout.addWidget(self.sidebar)

        # ==================================================
        # MAIN CHAT AREA
        # ==================================================

        chat_container = QWidget()
        chat_layout = QVBoxLayout(chat_container)
        chat_container.setStyleSheet("""
            QWidget {
                background-color: #111111;
                color: #E6E6E6;
            }
        """)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(0)

        # ==================================================
        # SIDEBAR TOGGLE
        # ==================================================

        # This is intentionally a small floating button rather than a
        # traditional top bar. The rest of the chat stays full-height.
        self.menu_button = QPushButton("☰", central)
        self.menu_button.setFixedSize(40, 40)
        self.menu_button.clicked.connect(self.toggle_sidebar)
        self.menu_button.setStyleSheet("""
            QPushButton {
                background-color: #111111;
                color: #D8D8D8;
                border: none;
                border-radius: 10px;
                font-size: 18px;
            }

            QPushButton:hover {
                background-color: #2A2A2A;
            }

            QPushButton:pressed {
                background-color: #333333;
            }
        """)
        self.menu_button.raise_()
        self.menu_button.move(16, 16)
        self.menu_button.raise_()

        # ==================================================
        # CHAT SCROLL AREA
        # ==================================================

        self.chat_scroll = QScrollArea()

        self.chat_scroll.setStyleSheet("""
            QScrollArea {
                background-color: #111111;
                border: none;
            }

            QScrollBar:vertical {
                background: #111111;
                width: 8px;
                margin: 0px;
            }

            QScrollBar::handle:vertical {
                background: #3A3A3A;
                border-radius: 4px;
                min-height: 30px;
            }

            QScrollBar::handle:vertical:hover {
                background: #505050;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setFrameShape(QFrame.NoFrame)

        self.chat_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )

        # Message container
        self.chat_widget = QWidget()
        self.chat_widget.setStyleSheet("background-color: #111111;")

        self.chat_layout = QVBoxLayout(self.chat_widget)

        self.chat_layout.setContentsMargins(
            40,
            25,
            40,
            25,
        )

        self.chat_layout.setSpacing(20)

        self.chat_layout.addStretch()

        self.chat_scroll.setWidget(self.chat_widget)

        self.auto_scroll = True
        self.chat_scroll.verticalScrollBar().valueChanged.connect(
            self._on_scroll_value_changed
        )

        chat_layout.addWidget(self.chat_scroll)

        # ==================================================
        # INPUT AREA
        # ==================================================

        input_container = QWidget()

        input_layout = QHBoxLayout(input_container)

        input_layout.setContentsMargins(16, 10, 16, 20)
        input_layout.setSpacing(10)
        input_layout.setAlignment(Qt.AlignBottom)

        self.attach_button = QPushButton("+")
        self.attach_button.setToolTip("Attach a file")
        self.attach_button.setFixedSize(45, 45)

        self.attach_button.setStyleSheet("""
            QPushButton {
                background-color: #2A2A2A;
                color: #E8E8E8;
                border: 1px solid #383838;
                border-radius: 12px;
                font-size: 22px;
            }

            QPushButton:hover {
                background-color: #333333;
            }

            QPushButton:pressed {
                background-color: #3A3A3A;
            }
        """)

        self.attach_button.clicked.connect(self.attach_file)

        self.input = MessageInput(
            self.send_message
        )
        self.input.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
    
        self.setStyleSheet("""
            QPlainTextEdit {
                background-color: #202020;
                color: #E8E8E8;
                border: 1px solid #343434;
                border-radius: 16px;
                padding: 12px 16px;
                font-size: 15px;
                selection-background-color: #3A3A3A;
                selection-color: #FFFFFF;
            }

            QPlainTextEdit:focus {
                border: 1px solid #454545;
            }

            QScrollBar:vertical {
                background: transparent;
                width: 6px;
                margin: 6px 3px;
            }

            QScrollBar::handle:vertical {
                background: #555555;
                border-radius: 3px;
                min-height: 25px;
            }

            QScrollBar::handle:vertical:hover {
                background: #666666;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }

            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)


        self.mic_button = QPushButton("🎤")
        self.mic_button.setToolTip("Voice input (coming soon)")
        self.mic_button.setEnabled(False)
        self.mic_button.setStyleSheet("""
            QPushButton {
                background-color: #2A2A2A;
                color: #8A8A8A;
                border: 1px solid #383838;
                border-radius: 12px;
                font-size: 17px;
            }
        """)
        self.mic_button.setFixedSize(45, 45)

        self.send_button = QPushButton("↑")
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #2A2A2A;
                color: #E8E8E8;
                border: 1px solid #383838;
                border-radius: 12px;
                font-size: 18px;
            }

            QPushButton:hover {
                background-color: #333333;
            }

            QPushButton:pressed {
                background-color: #3A3A3A;
            }

            QPushButton:disabled {
                color: #666666;
                background-color: #242424;
            }
        """)
        self.send_button.setFixedSize(45, 45)

        self.send_button.clicked.connect(
            self.send_message
        )

        input_layout.addWidget(self.attach_button, 0, Qt.AlignBottom)
        input_layout.addWidget(self.input, 1, Qt.AlignBottom)
        input_layout.addWidget(self.mic_button, 0, Qt.AlignBottom)
        input_layout.addWidget(self.send_button, 0, Qt.AlignBottom)

        chat_layout.addWidget(input_container)

        main_layout.addWidget(chat_container)


    def refresh_recent_chats(self):
        while self.recent_chats_layout.count():
            item = self.recent_chats_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for conversation_id, title, _updated_at, message_count, is_pinned in self.brain.memory.list_conversations():
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(3)

            display_title = ("📌 " if is_pinned else "") + (title or "New conversation")
            open_button = QPushButton(display_title)
            open_button.setFlat(True)
            open_button.setContextMenuPolicy(Qt.CustomContextMenu)
            open_button.customContextMenuRequested.connect(
                lambda position, cid=conversation_id, btn=open_button: self.show_conversation_menu(cid, btn.mapToGlobal(position))
            )
            open_button.setStyleSheet("""
                QPushButton { text-align: left; padding: 8px 6px; color: #dddddd;
                              background: transparent; border: none; border-radius: 7px; }
                QPushButton:hover { background: #222222; }
            """)
            open_button.clicked.connect(
                lambda checked=False, cid=conversation_id: self.open_conversation(cid)
            )
            row_layout.addWidget(open_button, 1)

            self.recent_chats_layout.addWidget(row)

        self.recent_chats_layout.addStretch()

    def show_conversation_menu(self, conversation_id, global_position):
        row = next(
            (item for item in self.brain.memory.list_conversations() if item[0] == conversation_id),
            None,
        )
        if row is None:
            self.refresh_recent_chats()
            return

        title = row[1] or "New conversation"
        is_pinned = bool(row[4])
        menu = QMenu(self)
        pin_action = menu.addAction("Unpin" if is_pinned else "Pin")
        rename_action = menu.addAction("Rename")
        menu.addSeparator()
        delete_action = menu.addAction("Delete")
        chosen = menu.exec(global_position)

        if chosen == pin_action:
            self.brain.memory.set_conversation_pinned(conversation_id, not is_pinned)
            self.refresh_recent_chats()
        elif chosen == rename_action:
            new_title, accepted = QInputDialog.getText(
                self, "Rename conversation", "Conversation name:", text=title
            )
            if accepted and new_title.strip():
                self.brain.memory.update_conversation_title(conversation_id, new_title)
                self.refresh_recent_chats()
        elif chosen == delete_action:
            reply = QMessageBox.question(
                self,
                "Delete conversation",
                f'Are you sure you want to delete "{title}"?\n\n'
                "This will permanently delete this conversation and its messages.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply != QMessageBox.Yes:
                return

            self.brain.memory.delete_conversation(conversation_id)

            if conversation_id == self.current_conversation_id:
                self.current_conversation_id = (
                    self.brain.memory.create_conversation()
                )
                self.chat_history = []
                self.brain.set_conversation(
                    self.current_conversation_id, []
                )
                self._clear_chat_display()

            self.refresh_recent_chats()

    def _clear_chat_display(self):
        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.chat_layout.addStretch()

    def new_chat(self):
        if not self.send_button.isEnabled():
            return
        self.current_conversation_id = self.brain.memory.create_conversation()
        self.chat_history = []
        self.brain.set_conversation(self.current_conversation_id, [])
        self._clear_chat_display()
        self.input.clear()
        self.pending_attachments = []
        self.attach_button.setText("+")
        self.attach_button.setToolTip("Attach an image")
        self.refresh_recent_chats()
        self.input.setFocus()

    def open_conversation(self, conversation_id):
        if not self.send_button.isEnabled():
            return
        saved_messages = self.brain.memory.get_conversation_messages(conversation_id)
        self.current_conversation_id = conversation_id
        self.chat_history = [
            {"role": role, "content": content} for role, content in saved_messages
        ]
        self.brain.set_conversation(conversation_id, saved_messages)
        self._clear_chat_display()
        self.input.clear()
        self.pending_attachments = []
        self.attach_button.setText("+")
        self.attach_button.setToolTip("Attach an image")
        for role, content in saved_messages:
            if role == "user":
                self.add_user_message(content)
            elif role == "assistant":
                self.add_jero_message(content)
        self.refresh_recent_chats()
        self.scroll_to_bottom()

    def toggle_sidebar(self):
        target_width = 0 if self.sidebar_open else 250

        self.sidebar_open = not self.sidebar_open

        self.sidebar_min_animation.stop()
        self.sidebar_max_animation.stop()

        for animation in (
            self.sidebar_min_animation,
            self.sidebar_max_animation,
        ):
            animation.setStartValue(self.sidebar.width())
            animation.setEndValue(target_width)

        self.sidebar_animation.start()

        # Keep the menu button visible when the sidebar is closed.
        self.menu_button.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if hasattr(self, "menu_button"):
            self.menu_button.move(16, 16)
            self.menu_button.raise_()

    # ======================================================
    # USER MESSAGE
    # ======================================================

    def add_user_message(self, message):
        row = QWidget()

        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        row_layout.addStretch()

        bubble = QFrame()

        bubble.setStyleSheet("""
            QFrame {
                background-color: #e5e5e5;
                border-radius: 14px;
            }
        """)

        bubble_layout = QVBoxLayout(bubble)

        bubble_layout.setContentsMargins(
            14,
            10,
            14,
            10,
        )

        label = QLabel(message)
        label.setWordWrap(True)

        label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        label.setStyleSheet("""
            QLabel {
                color: #111111;
                font-size: 14px;
            }
        """)

        bubble_layout.addWidget(label)

        row_layout.addWidget(bubble)

        self.chat_layout.insertWidget(
            self.chat_layout.count() - 1,
            row,
        )

        self.scroll_to_bottom()

    # ======================================================
    # JERO MESSAGE
    # ======================================================

    def add_jero_message(self, content):
        row = QWidget()

        row_layout = QHBoxLayout(row)

        row_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        response = QLabel()

        response.setWordWrap(True)
        response.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse
        )
        response.setTextFormat(Qt.RichText)

        response.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )

        response.setStyleSheet("""
            QLabel {
                background-color: transparent;
                color: #E6E6E6;
                font-size: 15px;
                padding: 0px;
            }
        """)

        html = render_markdown(content)

        response.setText(html)

        response.setStyleSheet("""
            QLabel {
                color: #E6E6E6;
                font-size: 15px;
            }
        """)

        row_layout.addWidget(response, 1)

        self.chat_layout.insertWidget(
            self.chat_layout.count() - 1,
            row,
        )

        return response

    def attach_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Attach an image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All Files (*.*)"
        )

        if not file_path:
            return

        suffix = Path(file_path).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            self.add_jero_message(
                "For now, JERO's local attachment support accepts image files "
                "(.png, .jpg, .jpeg, .webp, .bmp)."
            )
            return

        self.pending_attachments = [file_path]
        self.attach_button.setToolTip(f"Attached image: {Path(file_path).name}")
        self.attach_button.setText("✓")

    def send_message(self):
        message = self.input.toPlainText().strip()
        attachments = list(self.pending_attachments)

        if not message and not attachments:
            return

        if attachments and not message:
            message = "Describe the attached image."

        display_message = message
        if attachments:
            display_message += "\n[Image attached: " + ", ".join(
                Path(path).name for path in attachments
            ) + "]"

        is_first_message = not self.chat_history
        self.brain.memory.add_conversation_message(
            self.current_conversation_id, "user", display_message
        )
        if is_first_message:
            title = " ".join(message.split())[:45] or "Image conversation"
            self.brain.memory.update_conversation_title(
                self.current_conversation_id, title
            )
        self.refresh_recent_chats()
        self.add_user_message(display_message)

        self.chat_history.append(
            {
                "role": "user",
                "content": display_message,
            }
        )

        self.input.clear()
        self.auto_scroll = True
        self.pending_attachments = []
        self.attach_button.setText("+")
        self.attach_button.setToolTip("Attach an image")
        self.current_response = ""

        remembered = None if attachments else self.memory_handler.handle(message)

        if remembered is not None:
            response = "Got it, bro. I'll remember that."

            self.chat_history.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )
            self.brain.memory.add_conversation_message(
                self.current_conversation_id, "assistant", response
            )
            self.refresh_recent_chats()

            self.add_jero_message(response)

            self.input.setFocus()
            return

        self.send_button.setEnabled(False)
        self.input.setEnabled(False)

        self.thread = QThread()

        self.worker = GenerationWorker(
            self.brain,
            message,
            attachments=attachments,
        )

        self.worker.moveToThread(
            self.thread
        )

        self.thread.started.connect(
            self.worker.run
        )

        self.worker.chunk.connect(
            self.receive_chunk
        )

        self.worker.finished.connect(
            self.receive_finished
        )

        self.worker.error.connect(
            self.receive_error
        )

        self.worker.finished.connect(
            self.thread.quit
        )

        self.worker.error.connect(
            self.thread.quit
        )

        self.thread.finished.connect(
            self.worker.deleteLater
        )

        self.thread.finished.connect(
            self.thread.deleteLater
        )

        self.thread.start()

    # ======================================================
    # STREAMING RESPONSE
    # ======================================================

    def receive_chunk(self, chunk):
        if not self.current_response:
            self.current_response_label = QLabel()

            self.current_response_label.setWordWrap(
                True
            )

            self.current_response_label.setTextInteractionFlags(
                Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse
            )

            self.current_response_label.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Preferred,
            )

            self.current_response_label.setStyleSheet("""
                QLabel {
                    background-color: transparent;
                    color: #E6E6E6;
                    font-size: 15px;
                    padding: 0px;
                }
            """)

            row = QWidget()

            row_layout = QHBoxLayout(row)

            row_layout.setContentsMargins(
                0,
                0,
                0,
                0,
            )

            row_layout.addWidget(
                self.current_response_label,
                1,
            )

            self.chat_layout.insertWidget(
                self.chat_layout.count() - 1,
                row,
            )

        self.current_response += chunk

        self.current_response_label.setText(
            self.current_response
        )

        self.scroll_to_bottom_if_following()

    # ======================================================
    # RESPONSE FINISHED
    # ======================================================

    def receive_finished(self):
        self.chat_history.append(
            {
                "role": "assistant",
                "content": self.current_response,
            }
        )
        self.brain.memory.add_conversation_message(
            self.current_conversation_id, "assistant", self.current_response
        )
        self.refresh_recent_chats()

        if hasattr(
            self,
            "current_response_label",
        ):
            
            html = render_markdown(self.current_response)
            self.current_response_label.setText(html)


        self.send_button.setEnabled(True)
        self.input.setEnabled(True)
        self.input.setFocus()

        self.scroll_to_bottom()

    # ======================================================
    # ERROR
    # ======================================================

    def receive_error(self, error):
        self.add_jero_message(
            f"JERO ERROR: {error}"
        )

        self.send_button.setEnabled(True)
        self.input.setEnabled(True)
        self.input.setFocus()

    def closeEvent(self, event):
        # Stop only the local llama-server process launched by this JERO instance.
        try:
            self.brain.stop_server()
        except Exception:
            pass
        super().closeEvent(event)

    # ======================================================
    # SCROLL
    # ======================================================

    def scroll_to_bottom(self):
        scrollbar = (
            self.chat_scroll.verticalScrollBar()
        )

        scrollbar.setValue(
            scrollbar.maximum()
        )

    def _on_scroll_value_changed(self, value):
        sb = self.chat_scroll.verticalScrollBar()
        # Only stay "stuck" to the bottom if the user is at (or very near) the bottom
        self.auto_scroll = (sb.maximum() - value) <= 30

    def scroll_to_bottom_if_following(self):
        if self.auto_scroll:
            # Wait for the layout to grow before scrolling
            QTimer.singleShot(0, self.scroll_to_bottom)