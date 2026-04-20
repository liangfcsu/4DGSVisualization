"""
LeftControlPanel — collapsible parameter groups (Display, Gaussian, Camera, 4DGS Layers, Debug).
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QCheckBox,
    QScrollArea, QSizePolicy, QFrame,
)
from PyQt5.QtCore import Qt, pyqtSignal

from .state import UIState, RESOLUTION_OPTIONS, CAMERA_MODES, CAMERA_MODE_TO_INDEX
from .style import LEFT_PANEL_W, C_BORDER, SP_SM, SP_MD, SP_LG, S
from .widgets import CollapsibleSection, Separator


class LeftControlPanel(QWidget):
    """Left sidebar with collapsible parameter groups."""

    # Signals
    resolution_changed      = pyqtSignal(int, int)
    background_changed      = pyqtSignal(int)
    gamma_changed           = pyqtSignal(float)
    exposure_changed        = pyqtSignal(float)
    antialiasing_changed    = pyqtSignal(bool)
    point_size_changed      = pyqtSignal(float)
    splat_scale_changed     = pyqtSignal(float)
    alpha_scale_changed     = pyqtSignal(float)
    ring_size_changed       = pyqtSignal(float)
    show_centers_changed    = pyqtSignal(bool)
    show_ellipsoids_changed = pyqtSignal(bool)
    show_pointcloud_changed = pyqtSignal(bool)
    show_trails_changed     = pyqtSignal(bool)
    camera_mode_changed     = pyqtSignal(str)
    camera_selected         = pyqtSignal(int)
    fov_changed             = pyqtSignal(float)
    move_speed_changed      = pyqtSignal(float)
    rot_speed_changed       = pyqtSignal(float)
    reset_camera_clicked    = pyqtSignal()
    layer_changed           = pyqtSignal(str, bool)
    debug_changed           = pyqtSignal(str, bool)

    def __init__(self, state: UIState, cameras_info=None, parent=None):
        super().__init__(parent)
        self.state = state
        self.cameras_info = cameras_info or []
        self.setObjectName("LeftPanel")
        self.setFixedWidth(LEFT_PANEL_W())
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header
        header = QHBoxLayout()
        header.setContentsMargins(SP_MD(), SP_SM(), SP_SM(), SP_SM())
        title = QLabel("控制面板")
        title.setObjectName("PanelTitle")
        header.addWidget(title)
        header.addStretch()
        self._collapse_btn = QPushButton("◀")
        self._collapse_btn.setObjectName("TogglePanelBtn")
        self._collapse_btn.setFixedSize(S(20), S(24))
        self._collapse_btn.setToolTip("隐藏控制面板 (Tab)")
        self._collapse_btn.clicked.connect(lambda: self.setVisible(False))
        header.addWidget(self._collapse_btn)
        outer.addLayout(header)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_content = QWidget()
        self._layout = QVBoxLayout(scroll_content)
        self._layout.setContentsMargins(SP_SM(), 0, SP_SM(), SP_MD())
        self._layout.setSpacing(SP_SM())
        scroll.setWidget(scroll_content)
        outer.addWidget(scroll, stretch=1)

        self._build_display_section()
        self._build_gaussian_section()
        self._build_camera_section()
        self._build_layers_section()
        self._build_debug_section()
        self._layout.addStretch()

    # ── Display Section ───────────────────────────────────────────────────

    def _build_display_section(self):
        sec = CollapsibleSection("Display", expanded=True)
        cl = sec.content_layout

        # Resolution
        cl.addWidget(self._lbl("渲染分辨率"))
        self.res_combo = QComboBox()
        for label, *_ in RESOLUTION_OPTIONS:
            self.res_combo.addItem(label)
        self.res_combo.setToolTip("渲染分辨率 (1/2/3/4 快捷键)")
        self.res_combo.currentIndexChanged.connect(self._on_res_changed)
        cl.addWidget(self.res_combo)

        # Background
        cl.addWidget(self._lbl("背景色"))
        self.bg_combo = QComboBox()
        self.bg_combo.addItems(["黑色", "白色"])
        self.bg_combo.currentIndexChanged.connect(lambda i: self.background_changed.emit(i))
        cl.addWidget(self.bg_combo)

        # Gamma
        row = QHBoxLayout()
        row.addWidget(self._lbl("Gamma"))
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 5.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(self.state.gamma)
        self.gamma_spin.setToolTip("Gamma 校正值")
        self.gamma_spin.valueChanged.connect(lambda v: self.gamma_changed.emit(v))
        row.addWidget(self.gamma_spin)
        cl.addLayout(row)

        # Exposure
        row2 = QHBoxLayout()
        row2.addWidget(self._lbl("Exposure"))
        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setRange(0.1, 10.0)
        self.exposure_spin.setSingleStep(0.1)
        self.exposure_spin.setValue(self.state.exposure)
        self.exposure_spin.setToolTip("曝光补偿")
        self.exposure_spin.valueChanged.connect(lambda v: self.exposure_changed.emit(v))
        row2.addWidget(self.exposure_spin)
        cl.addLayout(row2)

        # Antialiasing
        self.aa_check = QCheckBox("抗锯齿")
        self.aa_check.setChecked(self.state.antialiasing)
        self.aa_check.setToolTip("抗锯齿")
        self.aa_check.toggled.connect(lambda v: self.antialiasing_changed.emit(v))
        cl.addWidget(self.aa_check)

        self._layout.addWidget(sec)

    # ── Gaussian Section ──────────────────────────────────────────────────

    def _build_gaussian_section(self):
        sec = CollapsibleSection("Gaussian", expanded=True)
        cl = sec.content_layout

        # Point size
        row = QHBoxLayout()
        row.addWidget(self._lbl("点大小"))
        self.point_size_spin = QDoubleSpinBox()
        self.point_size_spin.setRange(0.25, 4.0)
        self.point_size_spin.setSingleStep(0.25)
        self.point_size_spin.setDecimals(2)
        self.point_size_spin.setValue(self.state.point_size)
        self.point_size_spin.setToolTip("点模式下的高斯尺寸倍率")
        self.point_size_spin.valueChanged.connect(lambda v: self.point_size_changed.emit(v))
        row.addWidget(self.point_size_spin)
        cl.addLayout(row)

        # Splat scale
        row2 = QHBoxLayout()
        row2.addWidget(self._lbl("Splat Scale"))
        self.splat_scale_spin = QDoubleSpinBox()
        self.splat_scale_spin.setRange(0.1, 5.0)
        self.splat_scale_spin.setSingleStep(0.1)
        self.splat_scale_spin.setValue(self.state.splat_scale)
        self.splat_scale_spin.setToolTip("Splat 缩放因子")
        self.splat_scale_spin.valueChanged.connect(lambda v: self.splat_scale_changed.emit(v))
        row2.addWidget(self.splat_scale_spin)
        cl.addLayout(row2)

        # Alpha scale
        row3 = QHBoxLayout()
        row3.addWidget(self._lbl("Alpha Scale"))
        self.alpha_scale_spin = QDoubleSpinBox()
        self.alpha_scale_spin.setRange(0.1, 5.0)
        self.alpha_scale_spin.setSingleStep(0.1)
        self.alpha_scale_spin.setValue(self.state.alpha_scale)
        self.alpha_scale_spin.setToolTip("全局不透明度缩放因子")
        self.alpha_scale_spin.valueChanged.connect(lambda v: self.alpha_scale_changed.emit(v))
        row3.addWidget(self.alpha_scale_spin)
        cl.addLayout(row3)

        # Ring Size
        row_ring = QHBoxLayout()
        row_ring.addWidget(self._lbl("Ring 宽度"))
        self.ring_size_spin = QDoubleSpinBox()
        self.ring_size_spin.setRange(0.05, 0.95)
        self.ring_size_spin.setSingleStep(0.05)
        self.ring_size_spin.setDecimals(2)
        self.ring_size_spin.setValue(self.state.ring_size)
        self.ring_size_spin.setToolTip("Ring 模式环形宽度 (0.05=细环, 0.95=粗环)")
        self.ring_size_spin.valueChanged.connect(lambda v: self.ring_size_changed.emit(v))
        row_ring.addWidget(self.ring_size_spin)
        cl.addLayout(row_ring)

        # Checkboxes
        cl.addWidget(Separator())
        self.chk_centers = QCheckBox("显示高斯中心")
        self.chk_centers.setToolTip("显示高斯中心点")
        self.chk_centers.toggled.connect(lambda v: self.show_centers_changed.emit(v))
        cl.addWidget(self.chk_centers)

        self.chk_ellipsoids = QCheckBox("显示 Ellipsoid")
        self.chk_ellipsoids.setToolTip("显示高斯椭球")
        self.chk_ellipsoids.toggled.connect(lambda v: self.show_ellipsoids_changed.emit(v))
        cl.addWidget(self.chk_ellipsoids)

        self.chk_pointcloud = QCheckBox("显示 Point Cloud")
        self.chk_pointcloud.setToolTip("显示原始点云")
        self.chk_pointcloud.toggled.connect(lambda v: self.show_pointcloud_changed.emit(v))
        cl.addWidget(self.chk_pointcloud)

        self.chk_trails = QCheckBox("显示 Motion Trails")
        self.chk_trails.setToolTip("显示高斯运动轨迹")
        self.chk_trails.toggled.connect(lambda v: self.show_trails_changed.emit(v))
        cl.addWidget(self.chk_trails)

        self._layout.addWidget(sec)

    # ── Camera Section ────────────────────────────────────────────────────

    def _build_camera_section(self):
        sec = CollapsibleSection("Camera", expanded=False)
        cl = sec.content_layout

        # Camera mode
        cl.addWidget(self._lbl("相机模式"))
        self.cam_mode_combo = QComboBox()
        for label, _key in CAMERA_MODES:
            self.cam_mode_combo.addItem(label)
        self.cam_mode_combo.setToolTip("Y: Trackball, B: Orbit")
        self.cam_mode_combo.currentIndexChanged.connect(self._on_cam_mode)
        cl.addWidget(self.cam_mode_combo)

        # FOV
        row = QHBoxLayout()
        row.addWidget(self._lbl("FOV"))
        self.fov_spin = QDoubleSpinBox()
        self.fov_spin.setRange(10.0, 150.0)
        self.fov_spin.setSingleStep(5.0)
        self.fov_spin.setValue(60.0)
        self.fov_spin.setToolTip("视场角调节")
        self.fov_spin.valueChanged.connect(lambda v: self.fov_changed.emit(v))
        row.addWidget(self.fov_spin)
        cl.addLayout(row)

        # Move speed
        row_ms = QHBoxLayout()
        row_ms.addWidget(self._lbl("移动速度"))
        self.move_speed_spin = QDoubleSpinBox()
        self.move_speed_spin.setRange(0.001, 10.0)
        self.move_speed_spin.setSingleStep(0.1)
        self.move_speed_spin.setDecimals(3)
        self.move_speed_spin.setValue(0.5)
        self.move_speed_spin.setToolTip("相机移动速度")
        self.move_speed_spin.valueChanged.connect(lambda v: self.move_speed_changed.emit(v))
        row_ms.addWidget(self.move_speed_spin)
        cl.addLayout(row_ms)

        # Rotation speed
        row_rs = QHBoxLayout()
        row_rs.addWidget(self._lbl("旋转速度"))
        self.rot_speed_spin = QDoubleSpinBox()
        self.rot_speed_spin.setRange(0.001, 0.5)
        self.rot_speed_spin.setSingleStep(0.005)
        self.rot_speed_spin.setDecimals(3)
        self.rot_speed_spin.setValue(0.02)
        self.rot_speed_spin.setToolTip("相机旋转速度")
        self.rot_speed_spin.valueChanged.connect(lambda v: self.rot_speed_changed.emit(v))
        row_rs.addWidget(self.rot_speed_spin)
        cl.addLayout(row_rs)

        # Camera pose selector
        if self.cameras_info and len(self.cameras_info) > 0:
            cl.addWidget(Separator())
            cl.addWidget(self._lbl("相机位姿"))
            self.camera_pose_combo = QComboBox()
            self.camera_pose_combo.addItem("自由视角")
            for i, cam in enumerate(self.cameras_info):
                self.camera_pose_combo.addItem(f"{i+1}. {cam['name']}")
            self.camera_pose_combo.setToolTip("选择相机位姿 (N: 下一个, P: 最近)")
            self.camera_pose_combo.currentIndexChanged.connect(lambda i: self.camera_selected.emit(i))
            cl.addWidget(self.camera_pose_combo)
        else:
            self.camera_pose_combo = None

        cl.addWidget(Separator())
        btn_reset = QPushButton("↺ 重置相机")
        btn_reset.setToolTip("重置相机到初始位置 (R)")
        btn_reset.clicked.connect(self.reset_camera_clicked.emit)
        cl.addWidget(btn_reset)

        self._layout.addWidget(sec)

    # ── 4DGS Layers Section ───────────────────────────────────────────────

    def _build_layers_section(self):
        sec = CollapsibleSection("4DGS Layers", expanded=False)
        cl = sec.content_layout

        self._layer_checks = {}
        layers = [
            ("static",      "Static",            True),
            ("persistent",  "Persistent",         True),
            ("ephemeral",   "Ephemeral",          True),
            ("spawned",     "Spawned",            True),
            ("pruned",      "Pruned",             False),
            ("frustums",    "Camera Frustums",    False),
            ("centers",     "Gaussian Centers",   False),
            ("trails",      "Motion Trails",      False),
        ]
        for key, label, default in layers:
            chk = QCheckBox(label)
            chk.setChecked(default)
            chk.setToolTip(f"TODO: 切换 {label} 图层显隐")
            chk.toggled.connect(lambda v, k=key: self.layer_changed.emit(k, v))
            cl.addWidget(chk)
            self._layer_checks[key] = chk

        self._layout.addWidget(sec)

    # ── Debug Section ─────────────────────────────────────────────────────

    def _build_debug_section(self):
        sec = CollapsibleSection("Debug", expanded=False)
        cl = sec.content_layout

        self._debug_checks = {}
        options = [
            ("active_set",       "显示 Active Set"),
            ("visible_gauss",    "显示 Visible Gaussians"),
            ("cache_region",     "显示 Cache Region"),
            ("window_interval",  "显示 Window Interval"),
            ("diagnostics",      "Diagnostics Overlay"),
            ("bounding_boxes",   "Bounding Boxes"),
        ]
        for key, label in options:
            chk = QCheckBox(label)
            chk.setToolTip(f"TODO: {label}")
            chk.toggled.connect(lambda v, k=key: self.debug_changed.emit(k, v))
            cl.addWidget(chk)
            self._debug_checks[key] = chk

        self._layout.addWidget(sec)

    # ── Sync helpers ──────────────────────────────────────────────────────

    def sync_resolution(self, w: int, h: int):
        for i, (_, rw, rh) in enumerate(RESOLUTION_OPTIONS):
            if rw == w and rh == h:
                self.res_combo.blockSignals(True)
                self.res_combo.setCurrentIndex(i)
                self.res_combo.blockSignals(False)
                return

    def sync_camera_mode(self, mode_key: str):
        for i, (_label, key) in enumerate(CAMERA_MODES):
            if key == mode_key:
                self.cam_mode_combo.blockSignals(True)
                self.cam_mode_combo.setCurrentIndex(i)
                self.cam_mode_combo.blockSignals(False)
                return

    def sync_camera_pose(self, idx: int):
        """Sync camera pose combo. idx = -1 means free, otherwise camera index."""
        if self.camera_pose_combo:
            self.camera_pose_combo.blockSignals(True)
            self.camera_pose_combo.setCurrentIndex(idx + 1)
            self.camera_pose_combo.blockSignals(False)

    def sync_point_size(self, val: float):
        self.point_size_spin.blockSignals(True)
        self.point_size_spin.setValue(val)
        self.point_size_spin.blockSignals(False)

    # ── Internal ──────────────────────────────────────────────────────────

    def _on_res_changed(self, idx):
        if 0 <= idx < len(RESOLUTION_OPTIONS):
            _, w, h = RESOLUTION_OPTIONS[idx]
            self.resolution_changed.emit(w, h)

    def _on_cam_mode(self, idx):
        if 0 <= idx < len(CAMERA_MODES):
            _label, key = CAMERA_MODES[idx]
            self.camera_mode_changed.emit(key)

    @staticmethod
    def _lbl(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("SectionLabel")
        return lbl
