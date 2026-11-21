"""
Voice Chat Window - Separate window for hands-free voice conversation.

Features:
- Dark theme matching main app (#111111 background)
- Animated glowing orb with revolving arcs (QPainter, 60 FPS)
- State-reactive animation (Listening, Hearing, Thinking, Speaking, Standby)
- State label, live caption toggle, mute button, close button
"""

import math
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QRect, QSize,
    Signal, Slot, QObject
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QRadialGradient, QLinearGradient,
    QFont, QFontMetrics, QIcon, QPixmap, QPainterPath
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QApplication, QCheckBox
)

from jero.voice.state_machine import VoiceState
from jero.config import voice


class OrbWidget(QWidget):
    """
    Animated orb with revolving arcs.

    States:
    - LISTENING: Calm pulse, slow arc rotation
    - HEARING: Orb grows/brightens with volume, arcs react
    - THINKING: Arcs speed up, pulsing
    - SPEAKING: Orb pulses with audio output, arcs fast
    - STANDBY: Dimmed, arcs nearly still
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Animation state
        self._state = VoiceState.IDLE
        self._volume = 0.0          # Mic volume 0-1
        self._speaking_volume = 0.0 # TTS volume 0-1
        self._time = 0.0
        self._arc_rotation = 0.0

        # Colors
        self._orb_color = QColor("#00FF88")      # Bright green
        self._orb_glow_color = QColor("#00FF88")
        self._arc_color = QColor("#00FF88")
        self._arc_glow_color = QColor("#00FF88")
        self._standby_color = QColor("#444444")  # Dimmed
        self._standby_glow = QColor("#666666")

        # Animation timer (60 FPS)
        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / voice.ORB_ANIMATION_FPS))
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        # For smooth volume visualization
        self._volume_smooth = 0.0
        self._speaking_smooth = 0.0

    def set_state(self, state: VoiceState):
        """Set the current state."""
        self._state = state
        self.update()

    def set_volume(self, volume: float):
        """Set microphone volume level (0.0 - 1.0)."""
        self._volume = max(0.0, min(1.0, volume))

    def set_speaking_volume(self, volume: float):
        """Set TTS output volume level (0.0 - 1.0)."""
        self._speaking_volume = max(0.0, min(1.0, volume))

    def _tick(self):
        """Animation tick - called at 60 FPS."""
        dt = 1.0 / voice.ORB_ANIMATION_FPS
        self._time += dt

        # Smooth volume for visualization
        self._volume_smooth += (self._volume - self._volume_smooth) * 0.15
        self._speaking_smooth += (self._speaking_volume - self._speaking_smooth) * 0.15

        # Arc rotation speed based on state
        if self._state == VoiceState.LISTENING:
            arc_speed = voice.ORB_IDLE_PULSE_SPEED * 0.3
        elif self._state == VoiceState.HEARING:
            arc_speed = voice.ORB_HEARING_PULSE_SPEED * 0.5
        elif self._state == VoiceState.THINKING:
            arc_speed = voice.ORB_THINKING_ARC_SPEED
        elif self._state == VoiceState.SPEAKING:
            arc_speed = voice.ORB_SPEAKING_PULSE_SPEED * 0.7
        elif self._state == VoiceState.STANDBY:
            arc_speed = voice.ORB_STANDBY_ARC_SPEED
        else:
            arc_speed = 0.0

        self._arc_rotation += arc_speed * dt
        if self._arc_rotation > 2 * math.pi:
            self._arc_rotation -= 2 * math.pi

        self.update()

    def paintEvent(self, event):
        """Paint the orb and arcs."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # Center and radius
        cx = self.width() / 2
        cy = self.height() / 2
        base_radius = min(cx, cy) * 0.35

        # State-specific parameters
        if self._state == VoiceState.LISTENING:
            orb_radius = base_radius
            orb_alpha = 255
            arc_alpha = 180
            num_arcs = 3
            arc_width = 2
            arc_gap = 0.15

        elif self._state == VoiceState.HEARING:
            # Orb grows steadily with volume (no breathing sine oscillation)
            volume_boost = 1.0 + self._volume_smooth * 0.4
            orb_radius = base_radius * volume_boost
            orb_alpha = int(200 + 55 * self._volume_smooth)
            arc_alpha = int(200 + 55 * self._volume_smooth)
            num_arcs = 3
            arc_width = max(2, int(3 * volume_boost))
            arc_gap = 0.12

        elif self._state == VoiceState.THINKING:
            # Steady orb in thinking (no breathing sine oscillation)
            orb_radius = base_radius
            orb_alpha = 255
            arc_alpha = 220
            num_arcs = 3
            arc_width = 3
            arc_gap = 0.1

        elif self._state == VoiceState.SPEAKING:
            # Extremely subtle, low-amplitude micro-vibration when speaking
            vibe = (
                0.006 * math.sin(self._time * 24.0)
                + 0.004 * math.sin(self._time * 36.0)
            )
            orb_radius = base_radius * (1.0 + vibe)
            orb_alpha = 255
            arc_alpha = 230
            num_arcs = 3
            arc_width = 3
            arc_gap = 0.08

        elif self._state == VoiceState.STANDBY:
            # Dimmed, minimal animation
            orb_radius = base_radius * 0.85
            orb_alpha = 120
            arc_alpha = 60
            num_arcs = 3
            arc_width = 1
            arc_gap = 0.2

        else:  # IDLE
            orb_radius = base_radius * 0.7
            orb_alpha = 80
            arc_alpha = 40
            num_arcs = 3
            arc_width = 1
            arc_gap = 0.3

        # Draw arcs (revolving rings)
        self._draw_arcs(painter, cx, cy, orb_radius, num_arcs, arc_width, arc_gap, arc_alpha)

        # Draw orb (glowing sphere)
        self._draw_orb(painter, cx, cy, orb_radius, orb_alpha)

    def _draw_orb(self, painter: QPainter, cx: float, cy: float, radius: float, alpha: int):
        """Draw the central glowing orb."""
        # Outer glow
        glow_radius = radius * 1.8
        gradient = QRadialGradient(cx, cy, glow_radius)
        if self._state == VoiceState.STANDBY:
            gradient.setColorAt(0, QColor(self._standby_glow.red(), self._standby_glow.green(), self._standby_glow.blue(), alpha // 3))
            gradient.setColorAt(0.5, QColor(self._standby_glow.red(), self._standby_glow.green(), self._standby_glow.blue(), alpha // 6))
            gradient.setColorAt(1, QColor(self._standby_glow.red(), self._standby_glow.green(), self._standby_glow.blue(), 0))
        else:
            gradient.setColorAt(0, QColor(self._orb_glow_color.red(), self._orb_glow_color.green(), self._orb_glow_color.blue(), alpha // 2))
            gradient.setColorAt(0.5, QColor(self._orb_glow_color.red(), self._orb_glow_color.green(), self._orb_glow_color.blue(), alpha // 4))
            gradient.setColorAt(1, QColor(self._orb_glow_color.red(), self._orb_glow_color.green(), self._orb_glow_color.blue(), 0))

        painter.setPen(Qt.NoPen)
        painter.setBrush(gradient)
        painter.drawEllipse(QPoint(int(cx), int(cy)), int(glow_radius), int(glow_radius))

        # Inner glow (brighter core)
        core_radius = radius * 1.2
        gradient2 = QRadialGradient(cx, cy, core_radius)
        if self._state == VoiceState.STANDBY:
            gradient2.setColorAt(0, QColor(self._standby_color.red(), self._standby_color.green(), self._standby_color.blue(), alpha))
            gradient2.setColorAt(1, QColor(self._standby_color.red(), self._standby_color.green(), self._standby_color.blue(), alpha // 2))
        else:
            gradient2.setColorAt(0, QColor(self._orb_color.red(), self._orb_color.green(), self._orb_color.blue(), alpha))
            gradient2.setColorAt(1, QColor(self._orb_color.red(), self._orb_color.green(), self._orb_color.blue(), alpha // 2))

        painter.setBrush(gradient2)
        painter.drawEllipse(QPoint(int(cx), int(cy)), int(core_radius), int(core_radius))

        # Core orb
        gradient3 = QRadialGradient(cx, cy, radius)
        if self._state == VoiceState.STANDBY:
            gradient3.setColorAt(0, QColor(80, 80, 80, alpha))
            gradient3.setColorAt(0.7, QColor(60, 60, 60, alpha))
            gradient3.setColorAt(1, QColor(40, 40, 40, alpha))
        else:
            gradient3.setColorAt(0, QColor(0, 255, 136, alpha))
            gradient3.setColorAt(0.7, QColor(0, 200, 100, alpha))
            gradient3.setColorAt(1, QColor(0, 150, 80, alpha))

        painter.setBrush(gradient3)
        painter.drawEllipse(QPoint(int(cx), int(cy)), int(radius), int(radius))

        # Highlight (specular)
        highlight_radius = radius * 0.3
        hx = cx - radius * 0.3
        hy = cy - radius * 0.3
        gradient4 = QRadialGradient(hx, hy, highlight_radius)
        gradient4.setColorAt(0, QColor(255, 255, 255, alpha // 3))
        gradient4.setColorAt(1, QColor(255, 255, 255, 0))
        painter.setBrush(gradient4)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPoint(int(hx), int(hy)), int(highlight_radius), int(highlight_radius))

    def _draw_arcs(self, painter: QPainter, cx: float, cy: float, base_radius: float,
                   num_arcs: int, arc_width: int, arc_gap: float, alpha: int):
        """Draw revolving arcs around the orb revolving in alternating directions."""
        if self._state == VoiceState.IDLE:
            return

        painter.setPen(Qt.NoPen)

        for i in range(num_arcs):
            # Each arc at different radius
            arc_radius = base_radius * (1.3 + i * 0.28)

            # Arc angular span (in radians)
            arc_span = math.pi * 1.15  # ~207 degrees each for a clean balanced ring

            # Alternating rotation direction:
            # i = 0 (1st arc): clockwise (+1)
            # i = 1 (2nd arc): counter-clockwise (-1)
            # i = 2 (3rd arc): clockwise (+1)
            direction = 1.0 if (i % 2 == 0) else -1.0
            rotation = (self._arc_rotation * direction) + (i * arc_gap * 2 * math.pi)

            # Start angle
            start_angle = rotation

            # Color with alpha
            if self._state == VoiceState.STANDBY:
                color = QColor(self._standby_color.red(), self._standby_color.green(), self._standby_color.blue(), alpha)
            else:
                color = QColor(self._arc_color.red(), self._arc_color.green(), self._arc_color.blue(), alpha)

            # Convert to rectangle for drawArc
            rect_x = cx - arc_radius
            rect_y = cy - arc_radius
            rect_w = arc_radius * 2
            rect_h = arc_radius * 2

            pen = QPen(color, arc_width, Qt.SolidLine, Qt.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)

            # Qt drawArc: start_angle * 16, span_angle * 16 (in 1/16 degree)
            start_deg = int(start_angle * 180 / math.pi * 16)
            span_deg = int(arc_span * 180 / math.pi * 16)

            painter.drawArc(int(rect_x), int(rect_y), int(rect_w), int(rect_h), start_deg, span_deg)


class VoiceWindow(QWidget):
    """
    Main Voice Chat Window.

    Separate from main window, floats above or beside it.
    Dark theme, orb animation, state label, caption, mute, close.
    """

    # Signal to notify main window when closed
    closed = Signal()

    def __init__(self, brain, conversation_id: int, parent=None):
        super().__init__(parent, Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowTitle("JERO Voice")

        self.brain = brain
        self.conversation_id = conversation_id

        # Voice worker (will be started when window shows)
        self._voice_worker = None
        self._worker_thread = None

        # Caption state
        self._show_captions = True
        self._user_caption = ""
        self._jero_caption = ""

        # Drag support
        self._drag_pos = None

        self._setup_ui()
        self._apply_styles()

        # Window size
        self.resize(480, 580)

    def _setup_ui(self):
        """Setup the UI layout."""
        # Main container with rounded corners
        self.container = QFrame(self)
        self.container.setObjectName("voiceContainer")
        self.container.setFixedSize(440, 540)

        # Center the container
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(self.container, 0, Qt.AlignCenter)

        # Container layout
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # ========================================================
        # TITLE BAR (custom, draggable)
        # ========================================================
        title_bar = QFrame()
        title_bar.setObjectName("titleBar")
        title_bar.setFixedHeight(48)
        title_bar_layout = QHBoxLayout(title_bar)
        title_bar_layout.setContentsMargins(16, 0, 16, 0)

        # Title
        title_label = QLabel("JERO Voice")
        title_label.setObjectName("titleLabel")
        title_bar_layout.addWidget(title_label)

        title_bar_layout.addStretch()

        # Caption toggle
        self.caption_checkbox = QCheckBox("Captions")
        self.caption_checkbox.setObjectName("captionCheckbox")
        self.caption_checkbox.setChecked(True)
        self.caption_checkbox.toggled.connect(self._on_caption_toggled)
        title_bar_layout.addWidget(self.caption_checkbox)

        # Mute button
        self.mute_button = QPushButton("🎤")
        self.mute_button.setObjectName("muteButton")
        self.mute_button.setFixedSize(36, 36)
        self.mute_button.setCheckable(True)
        self.mute_button.setToolTip("Mute microphone")
        self.mute_button.clicked.connect(self._on_mute_toggled)
        title_bar_layout.addWidget(self.mute_button)

        # Close button
        self.close_button = QPushButton("✕")
        self.close_button.setObjectName("closeButton")
        self.close_button.setFixedSize(36, 36)
        self.close_button.setToolTip("Close voice chat")
        self.close_button.clicked.connect(self.close)
        title_bar_layout.addWidget(self.close_button)

        container_layout.addWidget(title_bar)

        # ========================================================
        # ORB AREA (center, expanding)
        # ========================================================
        self.orb_widget = OrbWidget()
        self.orb_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        container_layout.addWidget(self.orb_widget, 1)

        # ========================================================
        # STATE LABEL
        # ========================================================
        self.state_label = QLabel("Initializing...")
        self.state_label.setObjectName("stateLabel")
        self.state_label.setAlignment(Qt.AlignCenter)
        self.state_label.setFixedHeight(36)
        container_layout.addWidget(self.state_label)

        # ========================================================
        # CAPTION AREA
        # ========================================================
        self.caption_frame = QFrame()
        self.caption_frame.setObjectName("captionFrame")
        self.caption_frame.setFixedHeight(100)
        self.caption_frame.hide()  # Hidden by default, shown when captions enabled
        caption_layout = QVBoxLayout(self.caption_frame)
        caption_layout.setContentsMargins(16, 8, 16, 8)
        caption_layout.setSpacing(4)

        self.user_caption_label = QLabel("")
        self.user_caption_label.setObjectName("userCaption")
        self.user_caption_label.setWordWrap(True)
        self.user_caption_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        caption_layout.addWidget(self.user_caption_label)

        self.jero_caption_label = QLabel("")
        self.jero_caption_label.setObjectName("jeroCaption")
        self.jero_caption_label.setWordWrap(True)
        self.jero_caption_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        caption_layout.addWidget(self.jero_caption_label)

        caption_layout.addStretch()
        container_layout.addWidget(self.caption_frame)

        # ========================================================
        # BOTTOM SPACER
        # ========================================================
        container_layout.addSpacing(20)

    def _apply_styles(self):
        """Apply dark theme stylesheet."""
        self.setStyleSheet("""
            /* Main window background (translucent for rounded container) */
            VoiceWindow {
                background: transparent;
            }

            /* Container - rounded dark card */
            #voiceContainer {
                background-color: #111111;
                border: 1px solid #2A2A2A;
                border-radius: 20px;
            }

            /* Title bar */
            #titleBar {
                background: transparent;
                border-bottom: 1px solid #2A2A2A;
                border-top-left-radius: 20px;
                border-top-right-radius: 20px;
            }

            #titleLabel {
                color: #E6E6E6;
                font-size: 16px;
                font-weight: 600;
            }

            /* Buttons in title bar */
            #muteButton, #closeButton {
                background-color: #2A2A2A;
                color: #E6E6E6;
                border: 1px solid #383838;
                border-radius: 10px;
                font-size: 14px;
            }

            #muteButton:hover, #closeButton:hover {
                background-color: #333333;
            }

            #muteButton:pressed, #closeButton:pressed {
                background-color: #3A3A3A;
            }

            #muteButton:checked {
                background-color: #3A1A1A;
                border-color: #5A2A2A;
                color: #FF6666;
            }

            #captionCheckbox {
                color: #AAAAAA;
                font-size: 12px;
                spacing: 6px;
            }

            #captionCheckbox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                background-color: #2A2A2A;
            }

            #captionCheckbox::indicator:checked {
                background-color: #00FF88;
                border-color: #00FF88;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxMiAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEwIDNMNCA4TDIgNiIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPC9zdmc+);
            }

            /* State label */
            #stateLabel {
                color: #00FF88;
                font-size: 14px;
                font-weight: 500;
                letter-spacing: 1px;
                padding: 8px 0;
            }

            /* Caption frame */
            #captionFrame {
                background-color: #1A1A1A;
                border: 1px solid #2A2A2A;
                border-radius: 12px;
                margin: 0 16px 16px 16px;
            }

            #userCaption {
                color: #88CCFF;
                font-size: 13px;
                line-height: 1.4;
            }

            #jeroCaption {
                color: #00FF88;
                font-size: 13px;
                line-height: 1.4;
            }
        """)

    # ========================================================
    # Public Methods
    # ========================================================

    def start_voice(self):
        """Start the voice conversation worker."""
        from jero.voice.worker import VoiceWorkerThread

        self._worker_thread = VoiceWorkerThread(self.brain, self.conversation_id)
        self._voice_worker = self._worker_thread.worker

        # Connect signals
        self._voice_worker.state_changed.connect(self._on_state_changed)
        self._voice_worker.volume_level.connect(self._on_volume_level)
        self._voice_worker.user_transcript.connect(self._on_user_transcript)
        self._voice_worker.user_transcript_partial.connect(self._on_user_transcript_partial)
        self._voice_worker.jero_response.connect(self._on_jero_response)
        self._voice_worker.jero_response_chunk.connect(self._on_jero_response_chunk)
        self._voice_worker.error_occurred.connect(self._on_error)
        self._voice_worker.standby_activated.connect(self._on_standby_activated)
        self._voice_worker.wake_activated.connect(self._on_wake_activated)
        self._voice_worker.finished.connect(self._on_worker_finished)

        # Start worker thread
        self._worker_thread.start()

    def stop_voice(self):
        """Stop the voice conversation cleanly."""
        if self._worker_thread:
            if self._voice_worker:
                try:
                    self._voice_worker.state_changed.disconnect()
                    self._voice_worker.volume_level.disconnect()
                    self._voice_worker.user_transcript.disconnect()
                    self._voice_worker.user_transcript_partial.disconnect()
                    self._voice_worker.jero_response.disconnect()
                    self._voice_worker.jero_response_chunk.disconnect()
                    self._voice_worker.error_occurred.disconnect()
                    self._voice_worker.standby_activated.disconnect()
                    self._voice_worker.wake_activated.disconnect()
                    self._voice_worker.finished.disconnect()
                except Exception:
                    pass

            self._worker_thread.stop()
            self._worker_thread = None
            self._voice_worker = None

    def set_muted(self, muted: bool):
        """Mute/unmute microphone."""
        if self._voice_worker:
            if muted:
                self._voice_worker.mute()
            else:
                self._voice_worker.unmute()

    def interrupt(self):
        """Interrupt current speech."""
        if self._voice_worker:
            self._voice_worker.interrupt()

    # ========================================================
    # Signal Handlers
    # ========================================================

    @Slot(str)
    def _on_state_changed(self, state_label: str):
        """Update state label and orb animation."""
        self.state_label.setText(state_label)
        try:
            state = VoiceState[state_label.upper()]
            self.orb_widget.set_state(state)
        except KeyError:
            pass

    @Slot(float)
    def _on_volume_level(self, rms: float):
        """Update orb with mic volume."""
        self.orb_widget.set_volume(rms)

    @Slot(str)
    def _on_user_transcript(self, text: str):
        """Final user transcript."""
        self._user_caption = text
        self._update_captions()

    @Slot(str)
    def _on_user_transcript_partial(self, text: str):
        """Partial user transcript (while speaking)."""
        self._user_caption = text + "…"
        self._update_captions()

    @Slot(str)
    def _on_jero_response(self, text: str):
        """Full JERO response."""
        self._jero_caption = text
        self._update_captions()

    @Slot(str)
    def _on_jero_response_chunk(self, chunk: str):
        """Streaming JERO response chunk."""
        # For captions, we show the accumulated response
        # The orb speaking volume is driven by TTS audio output
        pass

    @Slot(str)
    def _on_error(self, msg: str):
        """Handle error."""
        print(f"[Voice] Error: {msg}")
        # Could show in UI temporarily

    @Slot()
    def _on_standby_activated(self):
        """Standby mode activated."""
        self.orb_widget.set_state(VoiceState.STANDBY)
        self.state_label.setText("Standby")

    @Slot()
    def _on_wake_activated(self):
        """Wake from standby."""
        pass  # State machine will transition to LISTENING

    @Slot()
    def _on_worker_finished(self):
        """Worker stopped."""
        pass

    def _update_captions(self):
        """Update caption labels."""
        if not self._show_captions:
            return

        self.user_caption_label.setText(f"You: {self._user_caption}" if self._user_caption else "")
        self.jero_caption_label.setText(f"JERO: {self._jero_caption}" if self._jero_caption else "")

    @Slot(bool)
    def _on_caption_toggled(self, checked: bool):
        """Toggle caption visibility."""
        self._show_captions = checked
        self.caption_frame.setVisible(checked)
        if checked:
            self._update_captions()

    @Slot(bool)
    def _on_mute_toggled(self, checked: bool):
        """Mute/unmute button clicked."""
        self.set_muted(checked)
        self.mute_button.setText("🔇" if checked else "🎤")
        self.mute_button.setToolTip("Unmute microphone" if checked else "Mute microphone")

    # ========================================================
    # Window Events
    # ========================================================

    def showEvent(self, event):
        """Window shown - start voice."""
        super().showEvent(event)
        # Start voice after window is shown
        QTimer.singleShot(100, self.start_voice)

    def closeEvent(self, event):
        """Window closing - stop voice."""
        self.stop_voice()
        self.closed.emit()
        super().closeEvent(event)

    # Draggable window
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Only drag from title bar area (top 48px of container)
            pos = event.position().toPoint()
            container_pos = self.container.mapFrom(self, pos)
            if container_pos.y() <= 48:
                self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        event.accept()