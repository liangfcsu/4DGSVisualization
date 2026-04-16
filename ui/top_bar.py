"""
TopBar — application title, mode switches, and global actions.
"""

from PyQt5.QtWidgets import (
    QToolBar, QWidget, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSizePolicy, QAction,
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal

from .state import (
    UIState, VISUALIZATION_MODES, CAMERA_MODES, PROJECT_MODES,
)
from .style import C_ACCENT, C_TEXT2, F_CAPTION, S


class TopBar(QToolBar):
    """
    Top toolbar: app title | scene badge | mode switch | vis mode | camera mode | actions
    """

    vis_mode_changed    = pyqtSignal(str)   # visualization mode key
    camera_mode_changed = pyqtSignal(str)   # camera mode key
    project_mode_changed = pyqtSignal(str)  # 3DGS / 4DGS / Hybrid
    reset_clicked       = pyqtSignal()
    screenshot_clicked  = pyqtSignal()
    fullscreen_clicked  = pyqtSignal()
    settings_clicked    = pyqtSignal()

    def __init__(self, state: UIState, parent=None):
        super().__init__("TopBar", parent)
        self.state = state
        self.setObjectName("TopBar")
        self.setMovable(False)
        self.setIconSize(QSize(S(16), S(16)))

        self._build()

    def _build(self):
        # ── Left: title + scene badge + project mode ──
        title = QLabel("4DGS Viewer")
        title.setObjectName("AppTitle")
        self.addWidget(title)
        self.addSeparator()

        self.scene_badge = QLabel(self.state.scene_name)
        self.scene_badge.setObjectName("SceneBadge")
        self.addWidget(self.scene_badge)

        self.mode_badge = QLabel(self.state.project_mode)
        self.mode_badge.setObjectName("ModeBadge")
        self.addWidget(self.mode_badge)

        self.addSeparator()

        # ── Center: project mode | vis mode | camera mode ──
        self.addWidget(self._lbl("模式:"))
        self.project_combo = QComboBox()
        for pm in PROJECT_MODES:
            self.project_combo.addItem(pm)
        self.project_combo.setFixedWidth(S(80))
        self.project_combo.setToolTip("切换项目模式 (3DGS / 4DGS / Hybrid)")
        idx = PROJECT_MODES.index(self.state.project_mode) if self.state.project_mode in PROJECT_MODES else 1
        self.project_combo.setCurrentIndex(idx)
        self.project_combo.currentIndexChanged.connect(self._on_project_mode)
        self.addWidget(self.project_combo)

        self.addSeparator()

        self.addWidget(self._lbl("显示:"))
        self.vis_combo = QComboBox()
        for label, _key in VISUALIZATION_MODES:
            self.vis_combo.addItem(label)
        self.vis_combo.setFixedWidth(S(100))
        self.vis_combo.setToolTip("切换显示模式 (G 键循环)")
        self.vis_combo.currentIndexChanged.connect(self._on_vis_mode)
        self.addWidget(self.vis_combo)

        self.addSeparator()

        self.addWidget(self._lbl("相机:"))
        self.camera_combo = QComboBox()
        for label, _key in CAMERA_MODES:
            self.camera_combo.addItem(label)
        self.camera_combo.setFixedWidth(S(100))
        self.camera_combo.setToolTip("切换相机模式 (Y: Trackball, B: Orbit)")
        self.camera_combo.currentIndexChanged.connect(self._on_camera_mode)
        self.addWidget(self.camera_combo)

        # ── Right spacer ──
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.addWidget(spacer)

        # ── Right: actions ──
        btn_reset = QPushButton("↺ 重置")
        btn_reset.setToolTip("重置相机到初始位置 (R)")
        btn_reset.clicked.connect(self.reset_clicked.emit)
        self.addWidget(btn_reset)

        btn_shot = QPushButton("📷 截图")
        btn_shot.setObjectName("PrimaryButton")
        btn_shot.setToolTip("保存截图 (M)")
        btn_shot.clicked.connect(self.screenshot_clicked.emit)
        self.addWidget(btn_shot)

        btn_fs = QPushButton("⛶")
        btn_fs.setToolTip("全屏 (F11 / F)")
        btn_fs.setFixedWidth(S(34))
        btn_fs.clicked.connect(self.fullscreen_clicked.emit)
        self.addWidget(btn_fs)

    # ── Sync helpers ──────────────────────────────────────────────────────

    def sync_vis_mode(self, mode_key: str):
        for i, (_label, key) in enumerate(VISUALIZATION_MODES):
            if key == mode_key:
                self.vis_combo.blockSignals(True)
                self.vis_combo.setCurrentIndex(i)
                self.vis_combo.blockSignals(False)
                return

    def sync_camera_mode(self, mode_key: str):
        for i, (_label, key) in enumerate(CAMERA_MODES):
            if key == mode_key:
                self.camera_combo.blockSignals(True)
                self.camera_combo.setCurrentIndex(i)
                self.camera_combo.blockSignals(False)
                return

    def sync_project_mode(self, mode: str):
        idx = PROJECT_MODES.index(mode) if mode in PROJECT_MODES else 0
        self.project_combo.blockSignals(True)
        self.project_combo.setCurrentIndex(idx)
        self.project_combo.blockSignals(False)
        self.mode_badge.setText(mode)

    def set_scene_name(self, name: str):
        self.scene_badge.setText(name)

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_vis_mode(self, idx):
        if 0 <= idx < len(VISUALIZATION_MODES):
            _label, key = VISUALIZATION_MODES[idx]
            self.vis_mode_changed.emit(key)

    def _on_camera_mode(self, idx):
        if 0 <= idx < len(CAMERA_MODES):
            _label, key = CAMERA_MODES[idx]
            self.camera_mode_changed.emit(key)

    def _on_project_mode(self, idx):
        if 0 <= idx < len(PROJECT_MODES):
            mode = PROJECT_MODES[idx]
            self.mode_badge.setText(mode)
            self.project_mode_changed.emit(mode)

    @staticmethod
    def _lbl(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("SectionLabel")
        return lbl
