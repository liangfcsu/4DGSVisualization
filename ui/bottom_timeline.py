"""
BottomTimelineBar — playback controls + timeline slider + lightweight status.
"""

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QSlider, QDoubleSpinBox, QCheckBox,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal

from .state import UIState, FPS_PRESETS
from .style import SP_SM, SP_MD, C_TEXT2, F_CAPTION, S


class BottomTimelineBar(QWidget):
    """Bottom bar: transport controls | timeline | frame/speed/cache info."""

    play_toggled    = pyqtSignal(bool)
    seek_pressed    = pyqtSignal()
    seek_released   = pyqtSignal(int)
    seek_moved      = pyqtSignal(int)
    first_frame     = pyqtSignal()
    prev_frame      = pyqtSignal()
    next_frame      = pyqtSignal()
    last_frame      = pyqtSignal()
    fps_changed     = pyqtSignal(float)
    loop_toggled    = pyqtSignal(bool)

    def __init__(self, state: UIState, has_sequence: bool, num_frames: int = 1, playback_fps: float = 30.0, parent=None):
        super().__init__(parent)
        self.state = state
        self.has_sequence = has_sequence
        self.setObjectName("BottomBar")
        self._seeking = False

        self._build(num_frames, playback_fps)

    def _build(self, num_frames: int, playback_fps: float):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(SP_MD(), SP_SM(), SP_MD(), SP_SM())
        main_layout.setSpacing(4)

        if self.has_sequence:
            # ── Timeline row ──
            timeline_row = QHBoxLayout()
            timeline_row.setSpacing(SP_SM())

            self.lbl_cur = QLabel("1")
            self.lbl_cur.setObjectName("FrameCounter")
            self.lbl_cur.setFixedWidth(S(40))
            self.lbl_cur.setAlignment(Qt.AlignCenter)

            self.seek = QSlider(Qt.Horizontal)
            self.seek.setRange(0, max(0, num_frames - 1))
            self.seek.setToolTip("拖动跳转帧位置")
            self.seek.sliderPressed.connect(self._on_seek_pressed)
            self.seek.sliderReleased.connect(self._on_seek_released)
            self.seek.sliderMoved.connect(self._on_seek_moved)

            self.lbl_total = QLabel(str(num_frames))
            self.lbl_total.setObjectName("SectionLabel")
            self.lbl_total.setFixedWidth(S(40))
            self.lbl_total.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            timeline_row.addWidget(self.lbl_cur)
            timeline_row.addWidget(self.seek, stretch=1)
            timeline_row.addWidget(self.lbl_total)
            main_layout.addLayout(timeline_row)

        # ── Controls row ──
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(4)

        if self.has_sequence:
            # Transport buttons
            self.btn_first = self._transport_btn("⏮", "第一帧 (Home)")
            self.btn_prev  = self._transport_btn("⏪", "上一帧 (←)")
            self.btn_play  = self._transport_btn("▶", "播放/暂停 (Space)", checkable=True)
            self.btn_next  = self._transport_btn("⏩", "下一帧 (→)")
            self.btn_last  = self._transport_btn("⏭", "最后一帧 (End)")

            self.btn_first.clicked.connect(self.first_frame.emit)
            self.btn_prev.clicked.connect(self.prev_frame.emit)
            self.btn_play.toggled.connect(self._on_play_toggled)
            self.btn_next.clicked.connect(self.next_frame.emit)
            self.btn_last.clicked.connect(self.last_frame.emit)

            ctrl_row.addStretch()
            for b in (self.btn_first, self.btn_prev, self.btn_play, self.btn_next, self.btn_last):
                ctrl_row.addWidget(b)
            ctrl_row.addSpacing(16)

            # FPS spinner
            ctrl_row.addWidget(self._lbl("速度:"))
            self.fps_spin = QDoubleSpinBox()
            self.fps_spin.setRange(0.1, 240.0)
            self.fps_spin.setValue(playback_fps)
            self.fps_spin.setSingleStep(1.0)
            self.fps_spin.setDecimals(1)
            self.fps_spin.setFixedWidth(S(65))
            self.fps_spin.setToolTip("播放速度 FPS (−/+ 调整)")
            self.fps_spin.valueChanged.connect(lambda v: self.fps_changed.emit(v))
            ctrl_row.addWidget(self.fps_spin)
            ctrl_row.addWidget(self._lbl("fps"))

            # FPS presets
            ctrl_row.addSpacing(8)
            for fps_val in [15, 30, 60]:
                fb = QPushButton(str(fps_val))
                fb.setObjectName("ChipButton")
                fb.setFixedSize(S(32), S(24))
                fb.setToolTip(f"设置 {fps_val} FPS")
                fb.clicked.connect(lambda _, v=fps_val: self._set_fps_preset(v))
                ctrl_row.addWidget(fb)

            ctrl_row.addSpacing(12)

            # Loop toggle
            self.loop_check = QCheckBox("循环")
            self.loop_check.setChecked(True)
            self.loop_check.setToolTip("循环播放")
            self.loop_check.toggled.connect(lambda v: self.loop_toggled.emit(v))
            ctrl_row.addWidget(self.loop_check)

        ctrl_row.addStretch()

        # Right-side lightweight info
        self.info_frame = QLabel("—")
        self.info_frame.setStyleSheet(f"color: {C_TEXT2}; font-size: {F_CAPTION()}px;")
        ctrl_row.addWidget(self.info_frame)

        self.info_cache = QLabel("")
        self.info_cache.setStyleSheet(f"color: {C_TEXT2}; font-size: {F_CAPTION()}px;")
        ctrl_row.addWidget(self.info_cache)

        main_layout.addLayout(ctrl_row)

        # Height
        self.setFixedHeight(S(86) if self.has_sequence else S(40))

    # ── Update from main loop ─────────────────────────────────────────────

    def update_playback(self, current_frame: int, total_frames: int,
                        is_playing: bool, fps: float, cache_pct: str):
        if not self.has_sequence:
            return

        if not self._seeking:
            self.seek.blockSignals(True)
            self.seek.setValue(current_frame)
            self.seek.blockSignals(False)

        self.lbl_cur.setText(str(current_frame + 1))

        self.btn_play.blockSignals(True)
        self.btn_play.setChecked(is_playing)
        self.btn_play.setText("⏸" if is_playing else "▶")
        self.btn_play.blockSignals(False)

        state_icon = "▶" if is_playing else "⏸"
        self.info_frame.setText(f"{state_icon} {current_frame + 1}/{total_frames}  {fps:.0f}fps")
        self.info_cache.setText(cache_pct)

    # ── Internal ──────────────────────────────────────────────────────────

    def _on_play_toggled(self, checked: bool):
        self.play_toggled.emit(checked)

    def _on_seek_pressed(self):
        self._seeking = True
        self.seek_pressed.emit()

    def _on_seek_released(self):
        self._seeking = False
        self.seek_released.emit(self.seek.value())

    def _on_seek_moved(self, value: int):
        self.lbl_cur.setText(str(value + 1))
        self.seek_moved.emit(value)

    def _set_fps_preset(self, fps_val: int):
        self.fps_spin.blockSignals(True)
        self.fps_spin.setValue(fps_val)
        self.fps_spin.blockSignals(False)
        self.fps_changed.emit(float(fps_val))

    def sync_fps(self, fps: float):
        self.fps_spin.blockSignals(True)
        self.fps_spin.setValue(fps)
        self.fps_spin.blockSignals(False)

    @staticmethod
    def _transport_btn(text: str, tip: str, checkable=False) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("TransportButton")
        btn.setToolTip(tip)
        btn.setFixedSize(S(34), S(34))
        if checkable:
            btn.setCheckable(True)
        return btn

    @staticmethod
    def _lbl(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("SectionLabel")
        return lbl
