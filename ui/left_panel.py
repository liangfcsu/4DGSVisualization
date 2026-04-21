"""
LeftControlPanel — collapsible parameter groups with only active controls.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox,
    QScrollArea, QSizePolicy, QGridLayout,
)
from PyQt5.QtCore import Qt, pyqtSignal

from .state import UIState, RESOLUTION_OPTIONS, CAMERA_MODES
from .style import LEFT_PANEL_W, C_BORDER, SP_SM, SP_MD, SP_LG, S
from .widgets import CollapsibleSection, Separator


class LeftControlPanel(QWidget):
    """Left sidebar with collapsible parameter groups."""

    # Signals
    resolution_changed      = pyqtSignal(int, int)
    background_changed      = pyqtSignal(int)
    gamma_changed           = pyqtSignal(float)
    exposure_changed        = pyqtSignal(float)
    point_size_changed      = pyqtSignal(float)
    alpha_scale_changed     = pyqtSignal(float)
    ring_size_changed       = pyqtSignal(float)
    camera_mode_changed     = pyqtSignal(str)
    camera_selected         = pyqtSignal(int)
    fov_changed             = pyqtSignal(float)
    move_speed_changed      = pyqtSignal(float)
    rot_speed_changed       = pyqtSignal(float)
    reset_camera_clicked    = pyqtSignal()
    selection_mode_toggled  = pyqtSignal(bool)
    polygon_selection_toggled = pyqtSignal(bool)
    select_all_clicked      = pyqtSignal()
    clear_selection_clicked = pyqtSignal()
    clear_selection_rect_clicked = pyqtSignal()
    invert_selection_clicked = pyqtSignal()
    hide_selected_clicked   = pyqtSignal()
    unhide_all_clicked      = pyqtSignal()
    delete_selected_clicked = pyqtSignal()
    delete_unselected_clicked = pyqtSignal()
    restore_deleted_clicked = pyqtSignal()
    load_sequence_clicked   = pyqtSignal()
    load_camera_clicked     = pyqtSignal()
    training_source_selected = pyqtSignal(str)  # input.ply路径，立即加载替换空白点云
    start_training_clicked  = pyqtSignal(str, str, int)  # source_path, model_path, iterations
    stop_training_clicked   = pyqtSignal()

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

        self._build_file_section()
        self._build_training_section()
        self._build_display_section()
        self._build_gaussian_section()
        self._build_selection_section()
        self._build_camera_section()
        self._layout.addStretch()

    # ── File Loading Section ──────────────────────────────────────────────

    def _build_file_section(self):
        sec = CollapsibleSection("文件加载", expanded=False)
        cl = sec.content_layout

        # Load sequence button
        load_seq_btn = QPushButton("加载点云序列")
        load_seq_btn.setToolTip("选择PLY序列文件夹或单个PLY文件")
        load_seq_btn.clicked.connect(self.load_sequence_clicked.emit)
        cl.addWidget(load_seq_btn)

        # Load camera button
        load_cam_btn = QPushButton("加载相机参数")
        load_cam_btn.setToolTip("选择COLMAP sparse/0文件夹")
        load_cam_btn.clicked.connect(self.load_camera_clicked.emit)
        cl.addWidget(load_cam_btn)
        
        # Current files info
        self.current_seq_label = QLabel("序列: 未加载")
        self.current_seq_label.setWordWrap(True)
        self.current_seq_label.setObjectName("SectionLabel")
        cl.addWidget(self.current_seq_label)
        
        self.current_cam_label = QLabel("相机: 未加载")
        self.current_cam_label.setWordWrap(True)
        self.current_cam_label.setObjectName("SectionLabel")
        cl.addWidget(self.current_cam_label)

        self._layout.addWidget(sec)
    # ── Display Section ───────────────────────────────────────────────────

    def _build_display_section(self):
        sec = CollapsibleSection("Display", expanded=False)
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

        self._layout.addWidget(sec)

    # ── Training Section ──────────────────────────────────────────────────

    def _build_training_section(self):
        sec = CollapsibleSection("训练控制", expanded=False)
        cl = sec.content_layout

        # Source path
        cl.addWidget(self._lbl("训练数据路径"))
        self.training_source_path = QLabel("未选择")
        self.training_source_path.setWordWrap(True)
        self.training_source_path.setObjectName("SectionLabel")
        cl.addWidget(self.training_source_path)
        
        btn_select_source = QPushButton("选择训练数据")
        btn_select_source.setToolTip("选择包含images和sparse文件夹的训练数据目录")
        btn_select_source.clicked.connect(self._select_training_source)
        cl.addWidget(btn_select_source)
        
        cl.addWidget(Separator())
        
        # Output path
        cl.addWidget(self._lbl("输出路径"))
        self.training_output_path = QLabel("未选择")
        self.training_output_path.setWordWrap(True)
        self.training_output_path.setObjectName("SectionLabel")
        cl.addWidget(self.training_output_path)
        
        btn_select_output = QPushButton("选择输出目录")
        btn_select_output.setToolTip("选择训练结果保存目录")
        btn_select_output.clicked.connect(self._select_training_output)
        cl.addWidget(btn_select_output)
        
        cl.addWidget(Separator())
        
        # Iterations
        row_iter = QHBoxLayout()
        row_iter.addWidget(self._lbl("迭代次数"))
        self.training_iterations = QDoubleSpinBox()
        self.training_iterations.setRange(1000, 100000)
        self.training_iterations.setValue(30000)
        self.training_iterations.setDecimals(0)
        self.training_iterations.setSingleStep(1000)
        self.training_iterations.setToolTip("训练迭代次数")
        row_iter.addWidget(self.training_iterations)
        cl.addLayout(row_iter)
        
        cl.addWidget(Separator())
        
        # Training status
        self.training_status = QLabel("状态: 未开始")
        self.training_status.setObjectName("SectionLabel")
        cl.addWidget(self.training_status)
        
        self.training_progress = QLabel("进度: 0 / 0")
        self.training_progress.setObjectName("SectionLabel")
        cl.addWidget(self.training_progress)
        
        self.training_loss = QLabel("Loss: -")
        self.training_loss.setObjectName("SectionLabel")
        cl.addWidget(self.training_loss)
        
        self.training_points = QLabel("点数: -")
        self.training_points.setObjectName("SectionLabel")
        cl.addWidget(self.training_points)
        
        cl.addWidget(Separator())
        
        # Control buttons
        self.btn_start_training = QPushButton("▶ 开始训练")
        self.btn_start_training.setToolTip("开始3DGS训练")
        self.btn_start_training.clicked.connect(self._on_start_training)
        cl.addWidget(self.btn_start_training)
        
        self.btn_stop_training = QPushButton("⏹ 停止训练")
        self.btn_stop_training.setToolTip("停止训练")
        self.btn_stop_training.clicked.connect(self.stop_training_clicked.emit)
        self.btn_stop_training.setEnabled(False)
        cl.addWidget(self.btn_stop_training)

        self._layout.addWidget(sec)
        
        # Store paths
        self._training_source = None
        self._training_output = None
    
    def _select_training_source(self):
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择训练数据目录",
            "",
            QFileDialog.ShowDirsOnly
        )
        if dir_path:
            self._training_source = dir_path
            import os
            self.training_source_path.setText(os.path.basename(dir_path))
            
            # 【关键修复】立即发射信号加载input.ply替换空白点云
            input_ply = os.path.join(dir_path, "input.ply")
            if os.path.exists(input_ply):
                print(f"[训练数据] 检测到input.ply，发射信号通知player加载")
                self.training_source_selected.emit(input_ply)
            else:
                print(f"[训练数据] ⚠ 未找到input.ply: {input_ply}")
    
    def _select_training_output(self):
        from PyQt5.QtWidgets import QFileDialog
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录",
            "",
            QFileDialog.ShowDirsOnly
        )
        if dir_path:
            self._training_output = dir_path
            import os
            self.training_output_path.setText(os.path.basename(dir_path))
    
    def _on_start_training(self):
        if not self._training_source or not self._training_output:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "参数缺失", "请先选择训练数据路径和输出路径")
            return
        
        iterations = int(self.training_iterations.value())
        self.start_training_clicked.emit(self._training_source, self._training_output, iterations)
    
    def update_training_status(self, status: str, iteration: int = 0, total: int = 0, 
                              loss: float = 0.0, num_points: int = 0):
        """更新训练状态显示"""
        self.training_status.setText(f"状态: {status}")
        if total > 0:
            self.training_progress.setText(f"进度: {iteration} / {total}")
        if loss > 0:
            self.training_loss.setText(f"Loss: {loss:.6f}")
        if num_points > 0:
            self.training_points.setText(f"点数: {num_points:,}")
    
    def set_training_active(self, active: bool):
        """设置训练按钮状态"""
        self.btn_start_training.setEnabled(not active)
        self.btn_stop_training.setEnabled(active)

    # ── Gaussian Section ──────────────────────────────────────────────────

    def _build_gaussian_section(self):
        sec = CollapsibleSection("Gaussian", expanded=False)
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

        # Alpha scale
        row2 = QHBoxLayout()
        row2.addWidget(self._lbl("不透明度"))
        self.alpha_scale_spin = QDoubleSpinBox()
        self.alpha_scale_spin.setRange(0.1, 5.0)
        self.alpha_scale_spin.setSingleStep(0.1)
        self.alpha_scale_spin.setValue(self.state.alpha_scale)
        self.alpha_scale_spin.setToolTip("点模式下的全局不透明度倍率")
        self.alpha_scale_spin.valueChanged.connect(lambda v: self.alpha_scale_changed.emit(v))
        row2.addWidget(self.alpha_scale_spin)
        cl.addLayout(row2)

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

        self._layout.addWidget(sec)

    # ── Selection Section ────────────────────────────────────────────────

    def _build_selection_section(self):
        sec = CollapsibleSection("Selection", expanded=False)
        cl = sec.content_layout

        self.selection_mode_btn = QPushButton("启用选择模式")
        self.selection_mode_btn.setCheckable(True)
        self.selection_mode_btn.setToolTip("开启后，左键单击可点选，左键拖拽可框选，Shift=添加，Ctrl=移除 (V)")
        self.selection_mode_btn.toggled.connect(self.selection_mode_toggled.emit)
        cl.addWidget(self.selection_mode_btn)
        
        self.polygon_selection_btn = QPushButton("多边形选择")
        self.polygon_selection_btn.setCheckable(True)
        self.polygon_selection_btn.setToolTip("开启后，左键添加顶点（最多8个），右键完成。可拖动顶点编辑 (X)")
        self.polygon_selection_btn.toggled.connect(self.polygon_selection_toggled.emit)
        cl.addWidget(self.polygon_selection_btn)

        self.selection_summary = QLabel("已选 0  |  可见 0  |  隐藏 0  |  删除 0")
        self.selection_summary.setObjectName("SectionLabel")
        self.selection_summary.setWordWrap(True)
        cl.addWidget(self.selection_summary)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(SP_SM())
        grid.setVerticalSpacing(SP_SM())

        buttons = [
            ("全选", self.select_all_clicked.emit, "选择全部可见高斯 (Ctrl+A)"),
            ("清空", self.clear_selection_clicked.emit, "清空当前选择 (Ctrl+Shift+A)"),
            ("清框参考", self.clear_selection_rect_clicked.emit, "清除持续框参考，保留当前已选结果 (Shift+C)"),
            ("反选", self.invert_selection_clicked.emit, "反转当前可见高斯的选择状态 (Ctrl+I)"),
            ("隐藏选中", self.hide_selected_clicked.emit, "隐藏选中的高斯 (Shift+H)"),
            ("恢复隐藏", self.unhide_all_clicked.emit, "恢复所有被隐藏的高斯 (Shift+U)"),
            ("删除选中", self.delete_selected_clicked.emit, "删除选中的高斯 (Delete)"),
            ("反向删除", self.delete_unselected_clicked.emit, "删除未选中的高斯，保留选中部分 (Shift+Delete)"),
            ("恢复删除", self.restore_deleted_clicked.emit, "恢复所有被删除的高斯 (Shift+R)"),
        ]

        for idx, (label, callback, tip) in enumerate(buttons):
            btn = QPushButton(label)
            btn.setToolTip(tip)
            btn.clicked.connect(callback)
            grid.addWidget(btn, idx // 2, idx % 2)

        cl.addLayout(grid)
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
    
    def update_file_info(self, seq_path=None, cam_path=None):
        """更新文件加载信息显示"""
        if seq_path is not None:
            import os
            seq_name = os.path.basename(seq_path) if seq_path else "未加载"
            self.current_seq_label.setText(f"序列: {seq_name}")
        if cam_path is not None:
            import os
            cam_name = os.path.basename(os.path.dirname(cam_path)) if cam_path else "未加载"
            self.current_cam_label.setText(f"相机: {cam_name}")

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
    
    def update_file_info(self, seq_path=None, cam_path=None):
        """更新文件加载信息显示"""
        if seq_path is not None:
            import os
            seq_name = os.path.basename(seq_path) if seq_path else "未加载"
            self.current_seq_label.setText(f"序列: {seq_name}")
        if cam_path is not None:
            import os
            cam_name = os.path.basename(os.path.dirname(cam_path)) if cam_path else "未加载"
            self.current_cam_label.setText(f"相机: {cam_name}")

    def sync_point_size(self, val: float):
        self.point_size_spin.blockSignals(True)
        self.point_size_spin.setValue(val)
        self.point_size_spin.blockSignals(False)

    def sync_selection_mode(self, enabled: bool):
        self.selection_mode_btn.blockSignals(True)
        self.selection_mode_btn.setChecked(enabled)
        self.selection_mode_btn.setText("选择模式已开启" if enabled else "启用选择模式")
        self.selection_mode_btn.blockSignals(False)

    def sync_selection_stats(self, selected: int, visible: int, hidden: int, deleted: int):
        self.selection_summary.setText(
            f"已选 {selected}  |  可见 {visible}  |  隐藏 {hidden}  |  删除 {deleted}"
        )

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
