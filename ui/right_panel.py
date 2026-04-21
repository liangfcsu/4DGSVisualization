"""
RightInfoPanel — compact runtime information with only live, trustworthy stats.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSizePolicy,
)
from PyQt5.QtCore import Qt

from .state import UIState, VISUALIZATION_MODES, CAMERA_MODES
from .style import RIGHT_PANEL_W, SP_SM, SP_MD, S
from .widgets import InfoCard


class RightInfoPanel(QWidget):
    """Right sidebar showing scene stats and runtime info as cards."""

    def __init__(self, state: UIState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setObjectName("RightPanel")
        self.setFixedWidth(RIGHT_PANEL_W())
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header
        header = QHBoxLayout()
        header.setContentsMargins(SP_MD(), SP_SM(), SP_SM(), SP_SM())
        title = QLabel("信息面板")
        title.setObjectName("PanelTitle")
        header.addWidget(title)
        header.addStretch()
        self._collapse_btn = QPushButton("▶")
        self._collapse_btn.setObjectName("TogglePanelBtn")
        self._collapse_btn.setFixedSize(S(20), S(24))
        self._collapse_btn.setToolTip("隐藏信息面板 (Tab)")
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

        self._build_scene_card()
        self._build_performance_card()
        self._build_cache_card()
        self._build_activity_card()
        self._layout.addStretch()

    # ── Scene Info ────────────────────────────────────────────────────────

    def _build_scene_card(self):
        self.scene_card = InfoCard("SCENE INFO")
        self._v_scene_name = self.scene_card.add_row("数据集")
        self._v_project_mode = self.scene_card.add_row("模式")
        self._v_frame = self.scene_card.add_row("帧")
        self._v_camera_mode = self.scene_card.add_row("相机")
        self._v_display_mode = self.scene_card.add_row("显示")
        self._v_total_gauss = self.scene_card.add_row("高斯数")
        self._v_visible_gauss = self.scene_card.add_row("可见高斯")
        self._v_selected_gauss = self.scene_card.add_row("选中高斯")
        self._v_hidden_gauss = self.scene_card.add_row("隐藏高斯")
        self._v_deleted_gauss = self.scene_card.add_row("删除高斯")
        self._layout.addWidget(self.scene_card)

    # ── Performance ───────────────────────────────────────────────────────

    def _build_performance_card(self):
        self.perf_card = InfoCard("PERFORMANCE")
        self._v_viewer_fps = self.perf_card.add_row("Viewer FPS")
        self._v_render_res = self.perf_card.add_row("渲染分辨率")
        self._v_window_size = self.perf_card.add_row("窗口尺寸")
        self._v_gpu = self.perf_card.add_row("GPU")
        self._v_cpu = self.perf_card.add_row("CPU Workers")
        self._v_gpu_mem = self.perf_card.add_row("显存")
        self._layout.addWidget(self.perf_card)

    # ── Stream & Cache ────────────────────────────────────────────────────

    def _build_cache_card(self):
        self.cache_card = InfoCard("STREAM & CACHE")
        self._v_load_mode = self.cache_card.add_row("加载模式")
        self._v_cache_status = self.cache_card.add_row("缓存状态")
        self._v_cache_hit = self.cache_card.add_row("缓存命中率")
        self._v_io_status = self.cache_card.add_row("IO 状态")
        self._v_preload = self.cache_card.add_row("预加载")
        self._layout.addWidget(self.cache_card)

    # ── Activity ──────────────────────────────────────────────────────────

    def _build_activity_card(self):
        self.activity_card = InfoCard("ACTIVITY")
        self._v_last_event = self.activity_card.add_row("最近事件")
        self._layout.addWidget(self.activity_card)

    # ── Update all values ─────────────────────────────────────────────────

    def update_info(self, state: UIState, extra: dict = None):
        """Refresh all card values from current state. Call every frame or on change."""
        extra = extra or {}

        # Scene Info
        self._v_scene_name.setText(state.scene_name)
        self._v_project_mode.setText(state.project_mode)
        self._v_frame.setText(f"{state.current_frame}/{state.total_frames}")

        cam_label = "—"
        for label, key in CAMERA_MODES:
            if key == state.camera_mode:
                cam_label = label
                break
        self._v_camera_mode.setText(cam_label)

        vis_label = "—"
        for label, key in VISUALIZATION_MODES:
            if key == state.vis_mode:
                vis_label = label
                break
        self._v_display_mode.setText(vis_label)
        total_g = state.total_gaussians
        self._v_total_gauss.setText(f"{total_g:,}" if total_g > 0 else "—")
        self._v_visible_gauss.setText(f"{state.visible_gaussians:,}" if state.visible_gaussians > 0 else "0")
        self._v_selected_gauss.setText(f"{state.selected_gaussians:,}" if state.selected_gaussians > 0 else "0")
        self._v_hidden_gauss.setText(f"{state.hidden_gaussians:,}" if state.hidden_gaussians > 0 else "0")
        self._v_deleted_gauss.setText(f"{state.deleted_gaussians:,}" if state.deleted_gaussians > 0 else "0")

        # Performance
        self._v_viewer_fps.setText(f"{state.viewer_fps:.1f}")
        self._v_render_res.setText(f"{state.render_w}×{state.render_h}")
        self._v_window_size.setText(f"{state.window_w}×{state.window_h}")
        self._v_gpu.setText(state.gpu_name or "—")
        self._v_cpu.setText(str(state.cpu_count) if state.cpu_count else "—")
        gpu_mem_str = "—"
        if state.gpu_memory_used != "—":
            gpu_mem_str = f"{state.gpu_memory_used} / {state.gpu_memory_total}"
        self._v_gpu_mem.setText(gpu_mem_str)

        # Stream & Cache
        self._v_load_mode.setText(state.load_mode or "N/A")
        self._v_cache_status.setText(state.cache_status or "—")
        if state.cache_hit_rate > 0:
            self._v_cache_hit.setText(f"{state.cache_hit_rate:.0f}%")
        else:
            self._v_cache_hit.setText("—")
        self._v_io_status.setText(extra.get("io_status", "—"))
        self._v_preload.setText(extra.get("preload_status", "—"))
        self._v_last_event.setText(extra.get("last_event", "—"))

    def set_last_event(self, text: str):
        self._v_last_event.setText(text)
