#!/usr/bin/env python3
"""
4DGS Interactive Player — Qt UI 交互式播放器 (Refactored)

用法:
    python player.py /path/to/global_per_frame_ply
    python player.py data/1/window_000/per_frame_ply --render-resolution 4k --playback-fps 30
    python player.py data/1/window_000/per_frame_ply --load-mode stream --gpu-cache-size 4
    python player.py point_cloud.ply --render-resolution 2k
    python player.py <sequence_dir> --sparse data/sparse

快捷键:
    1/2/3/4      - 切换渲染分辨率 (720p/1080p/2K/4K)
    G            - 切换显示模式循环
    W/A/S/D      - 相机平移
    Q/E          - 上下
    I/K/J/L      - 旋转
    U/O          - 滚转
    空格          - 播放/暂停
    ←/→          - 上/下一帧
    Home/End     - 第一帧/最后一帧
    -/+          - 调整播放FPS
    Y            - 切换 Trackball 模式
    B            - 切换 Orbit 模式
    N            - 下一个相机位姿
    P            - 跳转到最近的相机位姿
    R            - 重置相机
    M            - 截图
    F / F11      - 全屏
    Tab          - 隐藏/显示侧栏
    H            - 隐藏/显示 HUD
    Ctrl+Enter   - 演示模式
    Esc          - 退出
"""

import sys
import os
import time
import argparse
import importlib.util
from datetime import datetime

import numpy as np
import torch

# ─── 第一步：加载核心模块 ──────────────────────────────────────────────────────
_CORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "mult-frame_free-resolution_visualization.py")
_spec = importlib.util.spec_from_file_location("viewer_core", _CORE_PATH)
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)

GaussianFrame      = _core.GaussianFrame
SequenceManager    = _core.SequenceManager
GaussianPointCloud = _core.GaussianPointCloud
GaussianRenderer   = _core.GaussianRenderer
InteractiveCamera  = _core.InteractiveCamera

# ─── 第二步：修正 Qt 插件路径 ─────────────────────────────────────────────────
try:
    import PyQt5 as _PyQt5_probe
    for _sub in ("Qt5", "Qt"):
        _d = os.path.join(os.path.dirname(os.path.abspath(_PyQt5_probe.__file__)), _sub, "plugins")
        if os.path.exists(_d):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _d
            break
    del _PyQt5_probe, _sub, _d
except Exception:
    pass

# ─── 第三步：导入 PyQt5 ──────────────────────────────────────────────────────
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QSlider, QComboBox, QDoubleSpinBox,
        QToolBar, QStatusBar, QAction, QMenu, QSizePolicy,
        QFrame, QMessageBox, QFileDialog, QShortcut,
    )
    from PyQt5.QtCore import Qt, QTimer, QSize
    from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QIcon, QKeySequence
except ImportError:
    print("错误: 请先安装 PyQt5:  pip install PyQt5")
    sys.exit(1)

# ─── 第四步：导入 UI 模块 ────────────────────────────────────────────────────
from ui.state import (
    UIState, RESOLUTION_OPTIONS,
    VISUALIZATION_MODES, VIS_TO_RENDER_MODE,
    CAMERA_MODES,
)
from ui.style import GLOBAL_QSS, build_qss, F_CAPTION
from ui.top_bar import TopBar
from ui.left_panel import LeftControlPanel
from ui.right_panel import RightInfoPanel
from ui.bottom_timeline import BottomTimelineBar
from ui.overlay_hud import ViewportOverlay
from ui.widgets import ToastNotification


# ─── 渲染模式映射 ─────────────────────────────────────────────────────────────
RENDER_MODE_MAP = {
    "splat":  GaussianRenderer.RENDER_MODE_SPLAT,
    "points": GaussianRenderer.RENDER_MODE_POINTS,
    "ring":   GaussianRenderer.RENDER_MODE_RING,
}

RENDER_MODE_REVERSE = {v: k for k, v in RENDER_MODE_MAP.items()}

# vis_mode → render_mode constant
def _vis_to_render_mode(vis_key: str):
    rm_str = VIS_TO_RENDER_MODE.get(vis_key)
    if rm_str and rm_str in RENDER_MODE_MAP:
        return RENDER_MODE_MAP[rm_str]
    return None

# render_mode constant → vis_key
def _render_mode_to_vis(render_mode) -> str:
    rm_str = RENDER_MODE_REVERSE.get(render_mode, "")
    for vis_key, rm in VIS_TO_RENDER_MODE.items():
        if rm == rm_str:
            return vis_key
    return "rgb"


def _recommended_io_workers() -> int:
    cpu_count = os.cpu_count() or 4
    return max(4, min(8, cpu_count))

CAMERA_MOTION_KEYS = {
    Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D,
    Qt.Key_Q, Qt.Key_E, Qt.Key_I, Qt.Key_K,
    Qt.Key_J, Qt.Key_L, Qt.Key_U, Qt.Key_O,
}

CAMERA_KEY_TO_MODE = {
    "free":      InteractiveCamera.MODE_FPS,
    "trackball": InteractiveCamera.MODE_TRACKBALL,
    "orbit":     InteractiveCamera.MODE_ORBIT,
}


# ═══════════════════════════════════════════════════════════════════════════════
# RenderView — 渲染显示控件（鼠标交互）
# ═══════════════════════════════════════════════════════════════════════════════
class RenderView(QWidget):
    """渲染区域控件：显示 GPU 渲染结果，捕获鼠标/键盘驱动 InteractiveCamera"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("RenderView")
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background-color: #000;")

        self.camera: InteractiveCamera = None
        self._pixmap: QPixmap = None
        self._image_buffer: np.ndarray | None = None
        self._scaled_pixmap: QPixmap | None = None
        self._scaled_size: QSize | None = None
        self._scaled_key = None
        self._image_rect = None
        self._interaction_callback = None
        self._selection_callback = None

        self._left  = False
        self._right = False
        self._mid   = False
        self._last_pos = None
        self._trackball_ratio = 0.75
        self._selection_mode = False
        self._selection_drag = False
        self._selection_start = None
        self._selection_current = None
        self._selection_overlay_points = []
        self._persistent_selection_rect = None

        # Overlay references (set by MainWindow)
        self._overlay = None
        self._toast = None

    def set_camera(self, camera: InteractiveCamera):
        self.camera = camera

    def set_interaction_callback(self, callback):
        self._interaction_callback = callback

    def set_selection_callback(self, callback):
        self._selection_callback = callback

    def set_selection_mode(self, enabled: bool):
        self._selection_mode = bool(enabled)
        self._selection_drag = False
        self._selection_start = None
        self._selection_current = None
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self.update()

    def set_selection_overlay_points(self, points):
        self._selection_overlay_points = list(points or [])
        self.update()

    def set_persistent_selection_rect(self, rect_norm):
        self._persistent_selection_rect = rect_norm
        self.update()

    def is_interacting(self):
        return self._left or self._right or self._mid

    def update_image(self, rgb: np.ndarray):
        """接受 (H, W, 3) uint8 RGB 数组"""
        self._image_buffer = np.ascontiguousarray(rgb)
        h, w = self._image_buffer.shape[:2]
        qi = QImage(self._image_buffer.data, w, h, w * 3, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qi)
        self._scaled_pixmap = None
        self._scaled_size = None
        self._scaled_key = None
        self.update()

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter, QPen
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        self._image_rect = None
        if self._pixmap:
            pixmap_key = self._pixmap.cacheKey()
            target_size = self.size()
            if (
                self._scaled_pixmap is None
                or self._scaled_size != target_size
                or self._scaled_key != pixmap_key
            ):
                self._scaled_pixmap = self._pixmap.scaled(
                    target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self._scaled_size = target_size
                self._scaled_key = pixmap_key
            scaled = self._scaled_pixmap
            x = (self.width()  - scaled.width())  // 2
            y = (self.height() - scaled.height()) // 2
            self._image_rect = (x, y, scaled.width(), scaled.height())
            painter.drawPixmap(x, y, scaled)

        if self._image_rect and self._selection_overlay_points:
            x0, y0, w, h = self._image_rect
            sx = w / max(1, self.render_width())
            sy = h / max(1, self.render_height())
            pen = QPen(QColor(255, 167, 38, 220))
            pen.setWidth(2)
            painter.setPen(pen)
            for px, py, radius in self._selection_overlay_points:
                cx = x0 + px * sx
                cy = y0 + py * sy
                rr = max(3.0, radius * 0.6 * min(sx, sy))
                painter.drawEllipse(int(cx - rr), int(cy - rr), int(rr * 2), int(rr * 2))

        if self._selection_drag and self._selection_start and self._selection_current:
            pen = QPen(QColor(255, 110, 64, 240))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(255, 110, 64, 40))
            x0 = min(self._selection_start[0], self._selection_current[0])
            y0 = min(self._selection_start[1], self._selection_current[1])
            w = abs(self._selection_start[0] - self._selection_current[0])
            h = abs(self._selection_start[1] - self._selection_current[1])
            painter.drawRect(int(x0), int(y0), int(w), int(h))
        elif self._selection_mode and self._image_rect and self._persistent_selection_rect:
            x0, y0, w, h = self._image_rect
            start_norm, end_norm = self._persistent_selection_rect
            rx0 = x0 + min(start_norm[0], end_norm[0]) * w
            ry0 = y0 + min(start_norm[1], end_norm[1]) * h
            rw = abs(end_norm[0] - start_norm[0]) * w
            rh = abs(end_norm[1] - start_norm[1]) * h
            pen = QPen(QColor(80, 200, 255, 230))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QColor(80, 200, 255, 28))
            painter.drawRect(int(rx0), int(ry0), int(rw), int(rh))

    def resizeEvent(self, event):
        self._scaled_pixmap = None
        self._scaled_size = None
        self._scaled_key = None
        super().resizeEvent(event)
        # Reposition overlays
        if self._overlay:
            self._overlay.reposition(self.width(), self.height())
        if self._toast:
            self._toast.reposition()

    def _in_center(self, pos):
        cx, cy = self.width() / 2, self.height() / 2
        r = min(self.width(), self.height()) / 2 * self._trackball_ratio
        return (pos.x() - cx) ** 2 + (pos.y() - cy) ** 2 < r * r

    def render_width(self):
        if self.camera:
            return max(1, int(self.camera.width))
        return max(1, self.width())

    def render_height(self):
        if self.camera:
            return max(1, int(self.camera.height))
        return max(1, self.height())

    def _clamp_to_image_rect(self, pos):
        if not self._image_rect:
            return None
        x0, y0, w, h = self._image_rect
        px = min(max(pos.x(), x0), x0 + w)
        py = min(max(pos.y(), y0), y0 + h)
        return (px, py)

    def _point_inside_image(self, pos):
        if not self._image_rect:
            return False
        x0, y0, w, h = self._image_rect
        return x0 <= pos.x() <= x0 + w and y0 <= pos.y() <= y0 + h

    def _widget_point_to_norm(self, point):
        if not self._image_rect:
            return None
        x0, y0, w, h = self._image_rect
        px = min(max(point[0], x0), x0 + w)
        py = min(max(point[1], y0), y0 + h)
        return (
            (px - x0) / max(1.0, float(w)),
            (py - y0) / max(1.0, float(h)),
        )

    def _selection_op(self, modifiers):
        if modifiers & Qt.ShiftModifier:
            return "add"
        if modifiers & Qt.ControlModifier:
            return "remove"
        return "set"

    def mousePressEvent(self, e):
        if not self.camera:
            return
        if (
            self._selection_mode
            and e.button() == Qt.LeftButton
            and self._selection_callback
            and self._point_inside_image(e.pos())
        ):
            self._selection_drag = True
            self._selection_start = self._clamp_to_image_rect(e.pos())
            self._selection_current = self._selection_start
            self._last_pos = None
            self.setFocus()
            self.update()
            e.accept()
            return
        b = e.button()
        if   b == Qt.LeftButton:   self._left  = True
        elif b == Qt.RightButton:  self._right = True
        elif b == Qt.MiddleButton: self._mid   = True
        self._last_pos = e.pos()
        self.setFocus()
        if self._interaction_callback:
            self._interaction_callback()

    def mouseReleaseEvent(self, e):
        if self._selection_drag and e.button() == Qt.LeftButton:
            start = self._selection_start
            end = self._clamp_to_image_rect(e.pos()) or self._selection_current
            self._selection_drag = False
            self._selection_start = None
            self._selection_current = None
            self.update()
            if start and end and self._selection_callback:
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                op = self._selection_op(e.modifiers())
                if abs(dx) < 4 and abs(dy) < 4:
                    point_norm = self._widget_point_to_norm(end)
                    if point_norm is not None:
                        self._selection_callback("point", op, point_norm)
                else:
                    start_norm = self._widget_point_to_norm(start)
                    end_norm = self._widget_point_to_norm(end)
                    if start_norm is not None and end_norm is not None:
                        self._selection_callback("rect", op, (start_norm, end_norm))
            e.accept()
            return
        b = e.button()
        if   b == Qt.LeftButton:   self._left  = False
        elif b == Qt.RightButton:  self._right = False
        elif b == Qt.MiddleButton: self._mid   = False
        if not (self._left or self._right or self._mid):
            self._last_pos = None
        if self._interaction_callback:
            self._interaction_callback()

    def mouseMoveEvent(self, e):
        if self._selection_drag:
            self._selection_current = self._clamp_to_image_rect(e.pos()) or self._selection_current
            self.update()
            e.accept()
            return
        if not self.camera or self._last_pos is None:
            return
        dx = e.pos().x() - self._last_pos.x()
        dy = e.pos().y() - self._last_pos.y()

        if self.camera.mode == InteractiveCamera.MODE_TRACKBALL:
            ic = self._in_center(self._last_pos)
            if self._left:
                if ic: self.camera.trackball_rotate(dx, dy)
                else:  self.camera.trackball_roll(dx)
            elif self._right:
                if ic: self.camera.trackball_pan(-dx, dy)
                else:  self.camera.trackball_zoom(-dy * 0.5)
        else:
            # FPS / Orbit 模式: 左键旋转, 右键平移(抓取式), 中键前后
            if self._left:
                self.camera.rotate_yaw(dx * 0.3)
                self.camera.rotate_pitch(-dy * 0.3)
            if self._right:
                self.camera.move_right(-dx * 0.1)
                self.camera.move_up(dy * 0.1)
            if self._mid:
                self.camera.move_forward(-dy * 0.1)

        self._last_pos = e.pos()
        if self._interaction_callback:
            self._interaction_callback()

    def wheelEvent(self, e):
        if self.camera:
            self.camera.zoom(1 if e.angleDelta().y() > 0 else -1)
            if self._interaction_callback:
                self._interaction_callback()

    def keyPressEvent(self, e):
        e.ignore()  # 交由主窗口处理


# ═══════════════════════════════════════════════════════════════════════════════
# MainWindow — 主窗口 (使用模块化 UI 组件)
# ═══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):

    def __init__(
        self,
        renderer:    GaussianRenderer,
        camera:      InteractiveCamera,
        seq:         SequenceManager | None,
        render_w:    int = 1280,
        render_h:    int = 720,
    ):
        super().__init__()
        # ── Core objects (不改动) ──
        self.renderer   = renderer
        self.camera     = camera
        self.seq        = seq
        self.pc         = renderer.pc
        self.render_w   = render_w
        self.render_h   = render_h
        self._preferred_frame_device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── Runtime state ──
        self._fps_avg      = 0.0
        self._last_t       = time.time()
        self._keys_held: set = set()
        self._needs_render = True
        self._shortcuts    = []
        self._last_event   = ""
        self._update_counter = 0
        self._pending_seek_preview = None
        self._sticky_rect_selection = None
        self._sticky_rect_selection_op = "set"

        # ── UI State ──
        self.ui_state = UIState()
        self.ui_state.scene_name = self._scene_name()
        self.ui_state.project_mode = "4DGS" if seq else "3DGS"
        self.ui_state.total_frames = seq.num_frames if seq else 1
        self.ui_state.render_w = render_w
        self.ui_state.render_h = render_h
        self.ui_state.vis_mode = _render_mode_to_vis(renderer.render_mode)
        self.ui_state.point_size = getattr(renderer, "point_size", 1.0)
        self.ui_state.cpu_count = os.cpu_count() or 1
        if torch.cuda.is_available():
            self.ui_state.gpu_name = torch.cuda.get_device_name(0)
        if seq:
            self.ui_state.load_mode = "gpu_stream" if getattr(seq, '_gpu_stream_mode', False) else seq.load_mode
            self.ui_state.playback_fps = seq.fps
            self.ui_state.loop_enabled = getattr(seq, "loop", True)

        self.setWindowTitle(f"4DGS Viewer · {self.ui_state.scene_name}")
        self.setFocusPolicy(Qt.StrongFocus)

        # ── Build UI ──
        self._build_menu()
        self._build_ui()
        self._connect_signals()
        self._install_shortcuts()
        self.setStyleSheet(build_qss())
        self._apply_initial_window_size()

        # Sync initial state
        self._sync_initial_state()

        # ── Timers ──
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._on_render)
        self._render_timer.start(8)          # ~120 Hz UI 轮询上限

        self._move_timer = QTimer(self)
        self._move_timer.timeout.connect(self._process_held_keys)
        self._move_timer.start(16)

    # ═══════════════════════════════════════════════════════════════════════
    # UI Construction
    # ═══════════════════════════════════════════════════════════════════════

    def _build_menu(self):
        mb = self.menuBar()

        fm = mb.addMenu("文件(&F)")
        a = QAction("截图 (&M)", self); a.setShortcut("M"); a.triggered.connect(self._screenshot)
        fm.addAction(a)
        a = QAction("导出当前编辑帧 PLY", self); a.triggered.connect(self._export_current_frame_ply)
        fm.addAction(a)
        fm.addSeparator()
        a = QAction("退出 (&Q)", self); a.setShortcut("Ctrl+Q"); a.triggered.connect(self.close)
        fm.addAction(a)

        em = mb.addMenu("编辑(&E)")
        a = QAction("选择模式 (&V)", self); a.setShortcut("V"); a.triggered.connect(self._toggle_selection_mode)
        em.addAction(a)
        em.addSeparator()
        a = QAction("全选", self); a.setShortcut("Ctrl+A"); a.triggered.connect(self._select_all)
        em.addAction(a)
        a = QAction("清空选择", self); a.setShortcut("Ctrl+Shift+A"); a.triggered.connect(self._clear_selection)
        em.addAction(a)
        a = QAction("清除持续框", self); a.setShortcut("Shift+C"); a.triggered.connect(self._clear_selection_reference)
        em.addAction(a)
        a = QAction("反选", self); a.setShortcut("Ctrl+I"); a.triggered.connect(self._invert_selection)
        em.addAction(a)
        em.addSeparator()
        a = QAction("隐藏选中", self); a.setShortcut("Shift+H"); a.triggered.connect(self._hide_selected)
        em.addAction(a)
        a = QAction("恢复隐藏", self); a.setShortcut("Shift+U"); a.triggered.connect(self._unhide_all)
        em.addAction(a)
        a = QAction("删除选中", self); a.setShortcuts(["Delete", "Del", "Backspace"]); a.triggered.connect(self._delete_selected)
        em.addAction(a)
        a = QAction("恢复删除", self); a.setShortcut("Shift+R"); a.triggered.connect(self._restore_deleted)
        em.addAction(a)

        vm = mb.addMenu("视图(&V)")
        for label, w, h in RESOLUTION_OPTIONS:
            a = QAction(f"渲染: {label}", self)
            a.triggered.connect(lambda _, w=w, h=h: self._set_render_res(w, h))
            vm.addAction(a)
        vm.addSeparator()
        a = QAction("全屏 (&F11)", self); a.setShortcut("F11"); a.triggered.connect(self._toggle_fullscreen)
        vm.addAction(a)
        vm.addSeparator()
        a = QAction("隐藏/显示侧栏 (Tab)", self); a.triggered.connect(self._toggle_panels)
        vm.addAction(a)
        a = QAction("隐藏/显示 HUD (H)", self); a.triggered.connect(self._toggle_hud)
        vm.addAction(a)
        a = QAction("演示模式 (Ctrl+Enter)", self); a.triggered.connect(self._toggle_presentation)
        vm.addAction(a)

        hm = mb.addMenu("帮助(&H)")
        a = QAction("快捷键说明", self); a.triggered.connect(self._show_help)
        hm.addAction(a)

    def _build_ui(self):
        # ── Top Bar ──
        self.top_bar = TopBar(self.ui_state, self)
        self.addToolBar(Qt.TopToolBarArea, self.top_bar)

        # ── Central widget ──
        central = QWidget()
        central.setObjectName("CentralWidget")
        self.setCentralWidget(central)
        main_vl = QVBoxLayout(central)
        main_vl.setContentsMargins(0, 0, 0, 0)
        main_vl.setSpacing(0)

        # ── Middle section: left | viewport | right ──
        middle = QWidget()
        middle_hl = QHBoxLayout(middle)
        middle_hl.setContentsMargins(0, 0, 0, 0)
        middle_hl.setSpacing(0)

        # Left panel
        cameras_info = self.camera.cameras_info if hasattr(self.camera, 'cameras_info') else []
        self.left_panel = LeftControlPanel(self.ui_state, cameras_info=cameras_info)
        middle_hl.addWidget(self.left_panel)

        # Left toggle button (visible when panel hidden)
        self._left_toggle = QPushButton("▶")
        self._left_toggle.setObjectName("TogglePanelBtn")
        self._left_toggle.setToolTip("显示控制面板 (Tab)")
        self._left_toggle.clicked.connect(lambda: self._set_left_panel_visible(True))
        self._left_toggle.setVisible(False)
        middle_hl.addWidget(self._left_toggle)

        # Viewport
        self.view = RenderView()
        self.view.set_camera(self.camera)
        self.view.set_interaction_callback(self._request_render)
        self.view.set_selection_callback(self._handle_view_selection)
        middle_hl.addWidget(self.view, stretch=1)

        # Right toggle button (visible when panel hidden)
        self._right_toggle = QPushButton("◀")
        self._right_toggle.setObjectName("TogglePanelBtn")
        self._right_toggle.setToolTip("显示信息面板 (Tab)")
        self._right_toggle.clicked.connect(lambda: self._set_right_panel_visible(True))
        self._right_toggle.setVisible(False)
        middle_hl.addWidget(self._right_toggle)

        # Right panel
        self.right_panel = RightInfoPanel(self.ui_state)
        middle_hl.addWidget(self.right_panel)

        main_vl.addWidget(middle, stretch=1)

        # ── Bottom timeline ──
        self.bottom_bar = BottomTimelineBar(
            self.ui_state,
            has_sequence=self.seq is not None,
            num_frames=self.seq.num_frames if self.seq else 1,
            playback_fps=self.seq.fps if self.seq else 30.0,
        )
        main_vl.addWidget(self.bottom_bar)

        # ── Viewport overlays ──
        self.overlay = ViewportOverlay(self.view, self.ui_state)
        self.view._overlay = self.overlay

        self.toast = ToastNotification(self.view)
        self.view._toast = self.toast

        # ── Status bar (minimal) ──
        status_bar = QStatusBar()
        status_bar.setObjectName("AppStatusBar")
        status_bar.setStyleSheet(f"background: #0a1118; color: #5c7a94; border-top: 1px solid #1e3044; font-size: {F_CAPTION()}px;")
        self.setStatusBar(status_bar)

    def _sync_initial_state(self):
        """Sync all UI widgets to initial state."""
        self.top_bar.set_scene_name(self.ui_state.scene_name)
        self.top_bar.sync_project_mode(self.ui_state.project_mode)
        self.top_bar.sync_vis_mode(self.ui_state.vis_mode)
        self.top_bar.sync_camera_mode(self.ui_state.camera_mode)
        self.left_panel.sync_resolution(self.render_w, self.render_h)
        self.left_panel.sync_point_size(self.ui_state.point_size)
        self.left_panel.sync_selection_mode(self.ui_state.selection_mode)
        self.left_panel.sync_selection_stats(
            self.pc.selection_count,
            self.pc.visible_count,
            self.pc.hidden_count,
            self.pc.deleted_count,
        )
        # Sync camera params
        self.left_panel.move_speed_spin.blockSignals(True)
        self.left_panel.move_speed_spin.setValue(self.camera.move_speed)
        self.left_panel.move_speed_spin.blockSignals(False)
        self.left_panel.rot_speed_spin.blockSignals(True)
        self.left_panel.rot_speed_spin.setValue(self.camera.rot_speed)
        self.left_panel.rot_speed_spin.blockSignals(False)
        if self.seq:
            self.bottom_bar.sync_loop(self.ui_state.loop_enabled)
        self.overlay.sync_vis_mode(self.ui_state.vis_mode)
        self.overlay.set_selection_mode(self.ui_state.selection_mode)
        cam_label = dict((k, l) for l, k in CAMERA_MODES).get(self.ui_state.camera_mode, "Free")
        self.overlay.update_scene_label(
            self.ui_state.scene_name, self.ui_state.vis_mode, cam_label
        )
        self.view.set_selection_mode(self.ui_state.selection_mode)
        self.view.set_persistent_selection_rect(self._sticky_rect_selection)

    # ═══════════════════════════════════════════════════════════════════════
    # Signal Connections
    # ═══════════════════════════════════════════════════════════════════════

    def _connect_signals(self):
        # ── TopBar ──
        self.top_bar.vis_mode_changed.connect(self._set_vis_mode)
        self.top_bar.camera_mode_changed.connect(self._set_camera_mode)
        self.top_bar.reset_clicked.connect(self._reset_camera)
        self.top_bar.screenshot_clicked.connect(self._screenshot)
        self.top_bar.fullscreen_clicked.connect(self._toggle_fullscreen)

        # ── Left Panel ──
        self.left_panel.resolution_changed.connect(self._set_render_res)
        self.left_panel.background_changed.connect(self._on_bg_changed)
        self.left_panel.point_size_changed.connect(self._on_point_size_changed)
        self.left_panel.gamma_changed.connect(self._on_gamma_changed)
        self.left_panel.exposure_changed.connect(self._on_exposure_changed)
        self.left_panel.alpha_scale_changed.connect(self._on_alpha_scale_changed)
        self.left_panel.ring_size_changed.connect(self._on_ring_size_changed)
        self.left_panel.camera_mode_changed.connect(self._set_camera_mode)
        self.left_panel.camera_selected.connect(self._on_camera_selected)
        self.left_panel.fov_changed.connect(self._on_fov_changed)
        self.left_panel.move_speed_changed.connect(self._on_move_speed_changed)
        self.left_panel.rot_speed_changed.connect(self._on_rot_speed_changed)
        self.left_panel.reset_camera_clicked.connect(self._reset_camera)
        self.left_panel.selection_mode_toggled.connect(self._set_selection_mode)
        self.left_panel.select_all_clicked.connect(self._select_all)
        self.left_panel.clear_selection_clicked.connect(self._clear_selection)
        self.left_panel.clear_selection_rect_clicked.connect(self._clear_selection_reference)
        self.left_panel.invert_selection_clicked.connect(self._invert_selection)
        self.left_panel.hide_selected_clicked.connect(self._hide_selected)
        self.left_panel.unhide_all_clicked.connect(self._unhide_all)
        self.left_panel.delete_selected_clicked.connect(self._delete_selected)
        self.left_panel.restore_deleted_clicked.connect(self._restore_deleted)

        # ── Bottom Bar ──
        if self.seq:
            self.bottom_bar.play_toggled.connect(self._on_play_toggled)
            self.bottom_bar.first_frame.connect(lambda: self._jump_to_frame(0))
            self.bottom_bar.prev_frame.connect(lambda: (self.seq.prev_frame(), self._reload()))
            self.bottom_bar.next_frame.connect(lambda: (self.seq.next_frame(), self._reload()))
            self.bottom_bar.last_frame.connect(lambda: self._jump_to_frame(self.seq.num_frames - 1))
            self.bottom_bar.seek_moved.connect(self._on_seek_moved)
            self.bottom_bar.seek_released.connect(self._on_seek_released)
            self.bottom_bar.fps_changed.connect(lambda v: (self.seq.set_fps(v), self._request_render()))
            self.bottom_bar.loop_toggled.connect(self._on_loop_toggled)

        # ── Viewport Overlay ──
        self.overlay.quick_vis_mode_clicked.connect(self._set_vis_mode)

    # ═══════════════════════════════════════════════════════════════════════
    # Shortcuts
    # ═══════════════════════════════════════════════════════════════════════

    def _register_shortcut(self, key, callback):
        sc = QShortcut(QKeySequence(key), self)
        sc.setContext(Qt.WindowShortcut)
        sc.activated.connect(callback)
        self._shortcuts.append(sc)

    def _install_shortcuts(self):
        if self.seq:
            self._register_shortcut(Qt.Key_Space, self._toggle_play)
            self._register_shortcut(Qt.Key_Left, lambda: (self.seq.prev_frame(), self._reload()))
            self._register_shortcut(Qt.Key_Right, lambda: (self.seq.next_frame(), self._reload()))
            self._register_shortcut(Qt.Key_Home, lambda: self._jump_to_frame(0))
            self._register_shortcut(Qt.Key_End, lambda: self._jump_to_frame(self.seq.num_frames - 1))
            self._register_shortcut(Qt.Key_Minus, lambda: self._adj_fps(-5.0))
            self._register_shortcut(Qt.Key_Equal, lambda: self._adj_fps(+5.0))
            self._register_shortcut(Qt.Key_Plus, lambda: self._adj_fps(+5.0))

        self._register_shortcut(Qt.Key_Escape, self.close)
        self._register_shortcut(Qt.Key_F11, self._toggle_fullscreen)
        self._register_shortcut(Qt.Key_F, self._toggle_fullscreen)
        self._register_shortcut(Qt.Key_G, self._cycle_vis_mode)
        self._register_shortcut(Qt.Key_M, self._screenshot)
        self._register_shortcut(Qt.Key_Y, lambda: self._set_camera_mode("trackball"))
        self._register_shortcut(Qt.Key_B, lambda: self._set_camera_mode("orbit"))
        self._register_shortcut(Qt.Key_1, lambda: self._shortcut_resolution(0))
        self._register_shortcut(Qt.Key_2, lambda: self._shortcut_resolution(1))
        self._register_shortcut(Qt.Key_3, lambda: self._shortcut_resolution(2))
        self._register_shortcut(Qt.Key_4, lambda: self._shortcut_resolution(3))
        self._register_shortcut(Qt.Key_N, self._next_camera)
        self._register_shortcut(Qt.Key_P, self._snap_to_nearest_camera)
        self._register_shortcut(Qt.Key_R, self._reset_camera)
        self._register_shortcut(Qt.Key_V, self._toggle_selection_mode)
        self._register_shortcut("Ctrl+A", self._select_all)
        self._register_shortcut("Ctrl+Shift+A", self._clear_selection)
        self._register_shortcut("Shift+C", self._clear_selection_reference)
        self._register_shortcut("Ctrl+I", self._invert_selection)
        self._register_shortcut("Shift+H", self._hide_selected)
        self._register_shortcut("Shift+U", self._unhide_all)
        self._register_shortcut("Delete", self._delete_selected)
        self._register_shortcut("Del", self._delete_selected)
        self._register_shortcut("Backspace", self._delete_selected)
        self._register_shortcut("Shift+R", self._restore_deleted)
        self._register_shortcut(Qt.Key_Tab, self._toggle_panels)
        self._register_shortcut(Qt.Key_H, self._toggle_hud)
        self._register_shortcut("Ctrl+Return", self._toggle_presentation)

    # ═══════════════════════════════════════════════════════════════════════
    # Render Loop (核心逻辑不变)
    # ═══════════════════════════════════════════════════════════════════════

    def _request_render(self):
        self._needs_render = True

    def _set_sticky_rect_selection(self, rect_payload, op="set"):
        if rect_payload is None:
            self._sticky_rect_selection = None
            self._sticky_rect_selection_op = "set"
        else:
            start_norm, end_norm = rect_payload
            self._sticky_rect_selection = (
                (float(start_norm[0]), float(start_norm[1])),
                (float(end_norm[0]), float(end_norm[1])),
            )
            self._sticky_rect_selection_op = op
        self.view.set_persistent_selection_rect(self._sticky_rect_selection)

    def _clear_sticky_rect_selection(self):
        self._set_sticky_rect_selection(None, op="set")

    def _reapply_sticky_rect_selection(self):
        if not self._sticky_rect_selection:
            return False
        start_norm, end_norm = self._sticky_rect_selection
        indices = self.renderer.pick_rect(self.camera, start_norm, end_norm)
        self.pc.apply_selection_indices(indices, op=self._sticky_rect_selection_op)
        return True

    def _on_render(self):
        try:
            frame_updated = False

            # 序列推进 (不改动)
            if self.seq:
                self.seq.service_prefetch()
                if self._pending_seek_preview is not None and not self.seq.playing:
                    if self._apply_seek_preview_if_ready():
                        frame_updated = True
                if self.seq.playing:
                    due = self.seq.consume_due_frames()
                    if due > 0:
                        direction = 1 if self.seq.play_direction >= 0 else -1
                        if not self.seq.loop:
                            remaining = self.seq._remaining_steps_in_direction(direction=direction)
                            if remaining <= 0:
                                self.seq.playing = False
                                self._request_render()
                                due = 0
                            else:
                                due = min(due, remaining)

                    if due > 0:
                        ready_raw_idx = None
                        for offset in range(due, 0, -1):
                            raw_candidate = self.seq.current_frame + direction * offset
                            candidate = self.seq._normalize_frame_idx(raw_candidate)
                            if self.seq.is_frame_ready(candidate, prefer_device=self._preferred_frame_device):
                                ready_raw_idx = raw_candidate
                                break

                        if ready_raw_idx is not None:
                            self.seq.set_frame(ready_raw_idx)
                            self._reload()
                            frame_updated = True
                            if not self.seq.loop and self.seq.at_playback_boundary(direction=direction):
                                self.seq.playing = False
                        else:
                            target_idx = self.seq._normalize_frame_idx(self.seq.current_frame + direction * due)
                            self.seq.request_frame(target_idx, prefer_device=self._preferred_frame_device)

            if not (self._needs_render or frame_updated or self.view.is_interacting()):
                return

            # GPU 渲染 (不改动)
            rendered = self.renderer.render(self.camera)

            # Post-processing: gamma & exposure
            if self.ui_state.exposure != 1.0:
                rendered = rendered * self.ui_state.exposure
            if self.ui_state.gamma != 1.0:
                rendered = rendered.clamp(0, 1).pow(1.0 / self.ui_state.gamma)

            img = (
                rendered.mul(255.0)
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )

            self.view.update_image(img)
            self.view.set_selection_overlay_points(
                self.renderer.get_selection_overlay(self.camera)
                if self.pc.selection_count > 0 else []
            )
            self._needs_render = False
            self._update_ui()

        except Exception as e:
            self.statusBar().showMessage(f"渲染错误: {e}", 2000)

    def _reload(self):
        if self.seq:
            frame = self.seq.get_current_frame_data(prefer_device=self._preferred_frame_device)
            self.pc.apply_frame(frame)
            self._reapply_sticky_rect_selection()
            self._pending_seek_preview = None
        self._request_render()

    def _jump_to_frame(self, frame_idx: int):
        if not self.seq:
            return
        self.seq.play_direction = 1
        self.seq.set_frame(frame_idx, update_direction=False)
        self._reload()

    def _apply_seek_preview_if_ready(self) -> bool:
        if not self.seq or self._pending_seek_preview is None:
            return False
        target_idx = self._pending_seek_preview
        if not self.seq.is_frame_ready(target_idx, prefer_device=self._preferred_frame_device):
            return False
        frame = self.seq.get_current_frame_data(prefer_device=self._preferred_frame_device)
        self.pc.apply_frame(frame)
        self._reapply_sticky_rect_selection()
        self._pending_seek_preview = None
        self._request_render()
        return True

    # ═══════════════════════════════════════════════════════════════════════
    # UI Update (每帧刷新各面板)
    # ═══════════════════════════════════════════════════════════════════════

    def _update_ui(self):
        now = time.time()
        dt = max(now - self._last_t, 1e-6)
        self._last_t = now
        self._fps_avg = 0.1 * (1.0 / dt) + 0.9 * self._fps_avg
        self._update_counter += 1

        # Update state
        self.ui_state.viewer_fps = self._fps_avg
        self.ui_state.render_w = self.render_w
        self.ui_state.render_h = self.render_h
        self.ui_state.window_w = self.view.width()
        self.ui_state.window_h = self.view.height()
        self.ui_state.total_gaussians = self.pc.get_xyz.shape[0] if hasattr(self.pc, 'get_xyz') and self.pc.get_xyz is not None else 0
        self.ui_state.visible_gaussians = self.pc.visible_count
        self.ui_state.selected_gaussians = self.pc.selection_count
        self.ui_state.hidden_gaussians = self.pc.hidden_count
        self.ui_state.deleted_gaussians = self.pc.deleted_count

        cache_str = ""
        if self.seq:
            self.ui_state.current_frame = self.seq.current_frame + 1
            self.ui_state.total_frames = self.seq.num_frames
            self.ui_state.is_playing = self.seq.playing
            self.ui_state.playback_fps = self.seq.fps
            cache_str = self.seq.get_cache_status()
            self.ui_state.cache_status = cache_str
            stats = self.seq.get_load_stats()
            if stats['total_accesses'] > 0:
                self.ui_state.cache_hit_rate = stats['hit_rate']
            else:
                self.ui_state.cache_hit_rate = 0.0

        # GPU memory (not every frame - every 30 frames)
        if self._update_counter % 30 == 0 and torch.cuda.is_available():
            try:
                mem_used = torch.cuda.memory_allocated() / (1024**2)
                mem_total = torch.cuda.get_device_properties(0).total_mem / (1024**2)
                self.ui_state.gpu_memory_used = f"{mem_used:.0f}MB"
                self.ui_state.gpu_memory_total = f"{mem_total:.0f}MB"
            except Exception:
                pass

        # Update bottom bar
        if self.seq:
            self.bottom_bar.update_playback(
                self.seq.current_frame, self.seq.num_frames,
                self.seq.playing, self.seq.fps, cache_str
            )

        # Update overlay HUD
        frame_str = f"Frame {self.ui_state.current_frame}/{self.ui_state.total_frames}" if self.seq else "Static"
        cache_label = f"Cache: {cache_str}" if cache_str else ""
        self.overlay.update_hud(self._fps_avg, frame_str, cache_label)
        self.left_panel.sync_selection_stats(
            self.ui_state.selected_gaussians,
            self.ui_state.visible_gaussians,
            self.ui_state.hidden_gaussians,
            self.ui_state.deleted_gaussians,
        )

        # Update right panel (throttled to every 5 frames for performance)
        if self._update_counter % 5 == 0:
            preload_status = "—"
            io_status = f"×{_recommended_io_workers()}"
            if self.seq:
                io_status = f"×{self.seq.io_workers}"
                if self.seq.load_mode == SequenceManager.LOAD_MODE_PRELOAD_CPU:
                    if getattr(self.seq, "background_preload_completed", False):
                        preload_status = "完成"
                    else:
                        preload_status = f"{len(self.seq.cpu_cache)}/{self.seq.num_frames}"
                elif self.seq.load_mode == SequenceManager.LOAD_MODE_PRELOAD_GPU:
                    preload_status = f"{len(self.seq.gpu_cache)}/{self.seq.num_frames}"
                else:
                    preload_status = "按需预取"
            extra = {
                "io_status": io_status,
                "preload_status": preload_status,
                "last_event": self._last_event or "—",
            }
            self.right_panel.update_info(self.ui_state, extra)

        # Status bar (minimal)
        self.statusBar().showMessage(
            f"{self.render_w}×{self.render_h}  FPS: {self._fps_avg:.0f}  选中: {self.ui_state.selected_gaussians}"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Keyboard handling (不改动)
    # ═══════════════════════════════════════════════════════════════════════

    def keyPressEvent(self, e):
        k = e.key()
        modifiers = e.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier)
        if modifiers == Qt.NoModifier and k in CAMERA_MOTION_KEYS:
            self._keys_held.add(k)
            self._request_render()
            e.accept()
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        modifiers = e.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier)
        if modifiers == Qt.NoModifier and e.key() in CAMERA_MOTION_KEYS:
            self._keys_held.discard(e.key())
            self._request_render()
            e.accept()
            return
        super().keyReleaseEvent(e)

    def _process_held_keys(self):
        ks = self._keys_held
        if not ks:
            return
        moved = False
        if Qt.Key_W in ks: self.camera.move_forward(1); moved = True
        if Qt.Key_S in ks: self.camera.move_forward(-1); moved = True
        if Qt.Key_A in ks: self.camera.move_right(-1); moved = True
        if Qt.Key_D in ks: self.camera.move_right(1); moved = True
        if Qt.Key_Q in ks: self.camera.move_up(-1); moved = True
        if Qt.Key_E in ks: self.camera.move_up(1); moved = True
        if Qt.Key_I in ks: self.camera.rotate_pitch(1); moved = True
        if Qt.Key_K in ks: self.camera.rotate_pitch(-1); moved = True
        if Qt.Key_J in ks: self.camera.rotate_yaw(-1); moved = True
        if Qt.Key_L in ks: self.camera.rotate_yaw(1); moved = True
        if Qt.Key_U in ks: self.camera.rotate_roll(1); moved = True
        if Qt.Key_O in ks: self.camera.rotate_roll(-1); moved = True
        if moved:
            self._request_render()

    # ═══════════════════════════════════════════════════════════════════════
    # Actions — Visualization Mode
    # ═══════════════════════════════════════════════════════════════════════

    def _set_vis_mode(self, mode_key: str):
        """Set visualization mode."""
        render_mode = _vis_to_render_mode(mode_key)
        if render_mode is None:
            return
        self.renderer.set_render_mode(render_mode)
        self.ui_state.vis_mode = mode_key
        self.top_bar.sync_vis_mode(mode_key)
        self.overlay.sync_vis_mode(mode_key)
        vis_label = dict((k, l) for l, k in VISUALIZATION_MODES).get(mode_key, mode_key)
        self._set_last_event(f"显示: {vis_label}")
        self.toast.show_message(f"显示模式: {vis_label}", 1500)
        self._update_overlay_scene()
        self._request_render()

    def _cycle_vis_mode(self):
        """Cycle through implemented visualization modes (G key)."""
        implemented = [k for k in ["rgb", "gaussian", "ring"] if k in VIS_TO_RENDER_MODE]
        if not implemented:
            return
        try:
            idx = implemented.index(self.ui_state.vis_mode)
            next_idx = (idx + 1) % len(implemented)
        except ValueError:
            next_idx = 0
        self._set_vis_mode(implemented[next_idx])

    # ═══════════════════════════════════════════════════════════════════════
    # Actions — Camera Mode
    # ═══════════════════════════════════════════════════════════════════════

    def _set_camera_mode(self, mode_key: str):
        cam_mode = CAMERA_KEY_TO_MODE.get(mode_key)
        if cam_mode is not None:
            self.camera.switch_mode(cam_mode)
            self.ui_state.camera_mode = mode_key
            self.top_bar.sync_camera_mode(mode_key)
            self.left_panel.sync_camera_mode(mode_key)
            cam_label = dict((k, l) for l, k in CAMERA_MODES).get(mode_key, mode_key)
            self._set_last_event(f"相机: {cam_label}")
            self._update_overlay_scene()
            self._request_render()

    def _on_camera_selected(self, idx: int):
        """Camera pose selection from left panel."""
        if idx == 0:
            self.camera.current_camera_idx = -1
        else:
            camera_idx = idx - 1
            self.camera.set_camera(camera_idx)
        self._request_render()

    def _next_camera(self):
        if self.camera.cameras_info and len(self.camera.cameras_info) > 0:
            self.camera.next_camera()
            self.left_panel.sync_camera_pose(self.camera.current_camera_idx)
            self._request_render()

    def _snap_to_nearest_camera(self):
        if self.camera.cameras_info and len(self.camera.cameras_info) > 0:
            self.camera.snap_to_nearest_camera()
            self.left_panel.sync_camera_pose(self.camera.current_camera_idx)
            self._request_render()

    # ═══════════════════════════════════════════════════════════════════════
    # Actions — Resolution & Display
    # ═══════════════════════════════════════════════════════════════════════

    def _set_render_res(self, w: int, h: int):
        self.render_w = w
        self.render_h = h
        self.camera.resize(w, h)
        self.renderer._means2d_buffer = None
        self.left_panel.sync_resolution(w, h)
        self._set_last_event(f"分辨率: {w}×{h}")
        self.toast.show_message(f"渲染分辨率: {w}×{h}", 1500)
        self._request_render()

    def _shortcut_resolution(self, idx: int):
        if 0 <= idx < len(RESOLUTION_OPTIONS):
            _, w, h = RESOLUTION_OPTIONS[idx]
            self._set_render_res(w, h)

    def _on_bg_changed(self, idx: int):
        color = [1.0, 1.0, 1.0] if idx else [0.0, 0.0, 0.0]
        self.renderer.set_background_color(color)
        self._request_render()

    def _on_point_size_changed(self, value: float):
        self.renderer.set_point_style(size=value)
        self.ui_state.point_size = value
        self._request_render()

    def _on_gamma_changed(self, value: float):
        self.ui_state.gamma = value
        self._request_render()

    def _on_exposure_changed(self, value: float):
        self.ui_state.exposure = value
        self._request_render()

    def _on_alpha_scale_changed(self, value: float):
        self.ui_state.alpha_scale = value
        # Map alpha_scale to renderer point_opacity
        self.renderer.set_point_style(opacity=min(1.0, max(0.05, value)))
        self._request_render()

    def _on_ring_size_changed(self, value: float):
        self.ui_state.ring_size = value
        self.renderer.ring_size = value
        self._request_render()
        self._request_render()

    def _on_fov_changed(self, value_deg: float):
        import math
        self.camera.FoVx = math.radians(value_deg)
        # Recalculate FoVy from aspect ratio
        aspect = self.camera.width / max(1, self.camera.height)
        self.camera.FoVy = 2 * math.atan(math.tan(self.camera.FoVx / 2) / aspect)
        self._set_last_event(f"FOV: {value_deg:.0f}°")
        self._request_render()

    def _on_move_speed_changed(self, value: float):
        self.camera.move_speed = value
        self._set_last_event(f"移动速度: {value:.3f}")

    def _on_rot_speed_changed(self, value: float):
        self.camera.rot_speed = value
        self._set_last_event(f"旋转速度: {value:.3f}")

    def _on_loop_toggled(self, enabled: bool):
        self.ui_state.loop_enabled = enabled
        if self.seq:
            self.seq.loop = enabled
        self._set_last_event(f"循环播放: {'ON' if enabled else 'OFF'}")

    # ═══════════════════════════════════════════════════════════════════════
    # Actions — Selection
    # ═══════════════════════════════════════════════════════════════════════

    def _set_selection_mode(self, enabled: bool):
        enabled = bool(enabled)
        if not enabled:
            self._clear_selection_reference(quiet=True)
        self.ui_state.selection_mode = enabled
        self.left_panel.sync_selection_mode(enabled)
        self.overlay.set_selection_mode(enabled)
        self.view.set_selection_mode(enabled)
        self._set_last_event(f"选择模式: {'ON' if enabled else 'OFF'}")
        self.toast.show_message(
            "选择模式已开启：左键点选 / 拖拽框选 / Shift添加 / Ctrl移除"
            if enabled else "选择模式已关闭",
            1800,
        )
        self._request_render()

    def _toggle_selection_mode(self):
        self._set_selection_mode(not self.ui_state.selection_mode)

    def _selection_summary_text(self):
        return (
            f"已选 {self.pc.selection_count} / 可见 {self.pc.visible_count} / "
            f"隐藏 {self.pc.hidden_count} / 删除 {self.pc.deleted_count}"
        )

    def _apply_selection_feedback(self, prefix: str):
        summary = self._selection_summary_text()
        self._set_last_event(f"{prefix} · {summary}")
        self.toast.show_message(f"{prefix} · {summary}", 1800)
        self._request_render()

    def _handle_view_selection(self, kind: str, op: str, payload):
        if kind == "point":
            self._clear_sticky_rect_selection()
            idx = self.renderer.pick_point(self.camera, payload[0], payload[1])
            indices = [] if idx is None else [idx]
            action_label = "点选"
        else:
            start_norm, end_norm = payload
            self._set_sticky_rect_selection((start_norm, end_norm), op=op)
            indices = self.renderer.pick_rect(self.camera, start_norm, end_norm)
            action_label = "框选"
        self.pc.apply_selection_indices(indices, op=op)
        op_label = {"set": "设置", "add": "添加", "remove": "移除"}.get(op, op)
        extra = " · 已记住框区域并在换帧时自动沿用" if kind == "rect" else ""
        self._apply_selection_feedback(f"{action_label}{op_label}{extra}")

    def _clear_selection(self):
        self._clear_sticky_rect_selection()
        self.pc.clear_selection()
        self._apply_selection_feedback("清空选择")

    def _clear_selection_reference(self, quiet=False):
        had_rect = bool(self._sticky_rect_selection)
        self._clear_sticky_rect_selection()
        if quiet:
            return
        if had_rect:
            self._apply_selection_feedback("已清除持续框参考，当前选择结果已保留")
        else:
            self.toast.show_message("当前没有需要清除的持续框", 1500)

    def _select_all(self):
        self._clear_sticky_rect_selection()
        self.pc.select_all()
        self._apply_selection_feedback("全选可见高斯")

    def _invert_selection(self):
        self._clear_sticky_rect_selection()
        self.pc.invert_selection()
        self._apply_selection_feedback("反选完成")

    def _hide_selected(self):
        count = self.pc.hide_selected()
        if count <= 0:
            self.toast.show_message("当前没有可隐藏的选中高斯", 1500)
            return
        self._apply_selection_feedback(f"已隐藏 {count} 个高斯")

    def _unhide_all(self):
        count = self.pc.unhide_all()
        if count <= 0:
            self.toast.show_message("当前没有隐藏的高斯", 1500)
            return
        self._apply_selection_feedback(f"已恢复 {count} 个隐藏高斯")

    def _delete_selected(self):
        count = self.pc.delete_selected()
        if count <= 0:
            self.toast.show_message("当前没有可删除的选中高斯", 1500)
            return
        self._apply_selection_feedback(f"已删除 {count} 个高斯")

    def _restore_deleted(self):
        count = self.pc.restore_deleted()
        if count <= 0:
            self.toast.show_message("当前没有删除的高斯", 1500)
            return
        self._apply_selection_feedback(f"已恢复 {count} 个删除高斯")

    # ═══════════════════════════════════════════════════════════════════════
    # Actions — Playback
    # ═══════════════════════════════════════════════════════════════════════

    def _on_play_toggled(self, checked: bool):
        if self.seq:
            if checked:
                self.seq.play_direction = 1
            self.seq.playing = checked
            self.seq.last_update_time = time.time()
            self._reload()

    def _toggle_play(self):
        if self.seq:
            playing = self.seq.toggle_play()
            if playing:
                self.seq.play_direction = 1
            self._reload()

    def _on_seek_moved(self, value: int):
        if self.seq:
            self.seq.play_direction = 1
            self.seq.set_frame(value, update_direction=False)
            self.seq.request_frame(value, prefer_device=self._preferred_frame_device)
            self._pending_seek_preview = value
            self._apply_seek_preview_if_ready()

    def _on_seek_released(self, value: int):
        if self.seq:
            self.seq.play_direction = 1
            self.seq.set_frame(value, update_direction=False)
            self._reload()

    def _adj_fps(self, delta: float):
        if self.seq:
            new_fps = self.seq.adjust_fps(delta)
            self.bottom_bar.sync_fps(new_fps)
            self._request_render()

    # ═══════════════════════════════════════════════════════════════════════
    # Actions — Panel & HUD toggling
    # ═══════════════════════════════════════════════════════════════════════

    def _toggle_panels(self):
        """Toggle both left and right panels."""
        both_visible = self.left_panel.isVisible() and self.right_panel.isVisible()
        self._set_left_panel_visible(not both_visible)
        self._set_right_panel_visible(not both_visible)

    def _set_left_panel_visible(self, visible: bool):
        self.left_panel.setVisible(visible)
        self._left_toggle.setVisible(not visible)
        self.ui_state.left_panel_visible = visible

    def _set_right_panel_visible(self, visible: bool):
        self.right_panel.setVisible(visible)
        self._right_toggle.setVisible(not visible)
        self.ui_state.right_panel_visible = visible

    def _toggle_hud(self):
        self.ui_state.hud_visible = not self.ui_state.hud_visible
        self.overlay.set_hud_visible(self.ui_state.hud_visible)

    def _toggle_presentation(self):
        """Presentation mode: hide panels, minimize chrome, viewport + timeline + HUD only."""
        self.ui_state.presentation_mode = not self.ui_state.presentation_mode
        pres = self.ui_state.presentation_mode

        self._set_left_panel_visible(not pres)
        self._set_right_panel_visible(not pres)
        self.top_bar.setVisible(not pres)
        self.menuBar().setVisible(not pres)
        self.overlay.set_shortcuts_visible(not pres)

        if pres:
            self.toast.show_message("演示模式 — Ctrl+Enter 退出", 2000)
            self._set_last_event("进入演示模式")
        else:
            self._set_last_event("退出演示模式")

    # ═══════════════════════════════════════════════════════════════════════
    # Actions — Global
    # ═══════════════════════════════════════════════════════════════════════

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
        self._request_render()

    def _reset_camera(self):
        self.camera.reset()
        self._set_last_event("相机已重置")
        self.toast.show_message("相机已重置", 1500)
        self._request_render()

    def _screenshot(self):
        rendered = self.renderer.render(self.camera)
        img = rendered.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8).copy()
        h, w = img.shape[:2]
        qi = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qi)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"screenshot_{ts}.png"
        pixmap.save(fname)
        self._set_last_event(f"截图: {fname}")
        self.toast.show_message(f"截图已保存: {fname} ({w}×{h})", 3000)
        print(f"截图已保存: {fname}  ({w}×{h})")

    def _export_current_frame_ply(self):
        scene_base = self._scene_name().replace(os.sep, "_")
        default_name = f"{scene_base}_edited.ply"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出当前编辑后的 PLY",
            default_name,
            "PLY Files (*.ply)",
        )
        if not file_path:
            return
        try:
            self.pc.export_current_ply(file_path)
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", f"无法导出当前编辑帧:\n{exc}")
            return
        self._set_last_event(f"导出PLY: {os.path.basename(file_path)}")
        self.toast.show_message(f"已导出编辑后的 PLY: {os.path.basename(file_path)}", 2500)

    # ═══════════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _scene_name(self) -> str:
        if self.seq:
            return os.path.basename(os.path.abspath(self.seq.sequence_dir))
        if getattr(self.pc, "ply_path", None):
            return os.path.basename(self.pc.ply_path)
        return "Scene"

    def _set_last_event(self, text: str):
        self._last_event = text

    def _update_overlay_scene(self):
        cam_label = dict((k, l) for l, k in CAMERA_MODES).get(self.ui_state.camera_mode, "Free")
        self.overlay.update_scene_label(
            self.ui_state.scene_name, self.ui_state.vis_mode, cam_label
        )

    def _apply_initial_window_size(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1440, 900)
            return

        available = screen.availableGeometry()
        chrome_w = 72 + 260 + 280  # panels
        chrome_h = 200
        max_content_w = max(640, int(available.width() * 0.88) - chrome_w)
        max_content_h = max(360, int(available.height() * 0.85) - chrome_h)
        scale = min(
            max_content_w / max(1, self.render_w),
            max_content_h / max(1, self.render_h),
            1.0,
        )
        content_w = max(720, int(self.render_w * scale))
        content_h = max(405, int(self.render_h * scale))
        window_w = min(available.width(), content_w + chrome_w)
        window_h = min(available.height(), content_h + chrome_h)
        self.resize(window_w, window_h)
        self.move(
            available.x() + max(0, (available.width() - window_w) // 2),
            available.y() + max(0, (available.height() - window_h) // 2),
        )

    def _show_help(self):
        QMessageBox.information(self, "快捷键帮助", """
<b>分辨率控制</b><br>
&nbsp;&nbsp;<b>1/2/3/4</b> — 720p / 1080p / 2K / 4K<br><br>

<b>显示模式</b><br>
&nbsp;&nbsp;<b>G</b> — 循环切换 (RGB / Gaussian / Ring)<br><br>

<b>鼠标操作 (FPS/Orbit 模式)</b><br>
&nbsp;&nbsp;<b>左键拖动</b> — 旋转视角<br>
&nbsp;&nbsp;<b>右键拖动</b> — 平移<br>
&nbsp;&nbsp;<b>中键拖动</b> — 前后移动<br>
&nbsp;&nbsp;<b>滚轮</b> — 缩放<br><br>

<b>鼠标操作 (Trackball 模式)</b><br>
&nbsp;&nbsp;<b>左键中心</b> — 球面旋转 &nbsp; <b>左键边缘</b> — 滚转<br>
&nbsp;&nbsp;<b>右键中心</b> — 平移 &nbsp; <b>右键边缘</b> — 缩放<br><br>

<b>高斯选择 / 编辑</b><br>
&nbsp;&nbsp;<b>V</b> — 开关选择模式<br>
&nbsp;&nbsp;<b>左键单击</b> — 点选高斯 &nbsp; <b>左键拖拽</b> — 框选高斯<br>
&nbsp;&nbsp;框选区域会在换帧时自动沿用；点选 / 全选 / 清空 / 反选会取消沿用<br>
&nbsp;&nbsp;<b>Shift</b> — 添加到选择 &nbsp; <b>Ctrl</b> — 从选择中移除<br>
&nbsp;&nbsp;<b>Ctrl+A</b> — 全选 &nbsp; <b>Ctrl+Shift+A</b> — 清空选择 &nbsp; <b>Ctrl+I</b> — 反选<br>
&nbsp;&nbsp;<b>Delete/Del</b> — 删除选中 &nbsp; <b>Shift+C</b> — 清除持续框 &nbsp; <b>Shift+H</b> — 隐藏选中<br>
&nbsp;&nbsp;<b>Shift+U</b> — 恢复隐藏 &nbsp; <b>Shift+R</b> — 恢复删除<br><br>

<b>相机移动</b><br>
&nbsp;&nbsp;<b>W/S</b> 前后 &nbsp; <b>A/D</b> 左右 &nbsp; <b>Q/E</b> 上下<br>
&nbsp;&nbsp;<b>I/K</b> 俯仰 &nbsp; <b>J/L</b> 偏航 &nbsp; <b>U/O</b> 滚转<br>
&nbsp;&nbsp;<b>Y</b> Trackball &nbsp; <b>B</b> Orbit &nbsp; <b>R</b> 重置相机<br>
&nbsp;&nbsp;<b>N</b> 下一相机 &nbsp; <b>P</b> 跳转最近相机<br><br>

<b>播放</b><br>
&nbsp;&nbsp;<b>Space</b> 播放/暂停 &nbsp; <b>←/→</b> 逐帧 &nbsp; <b>Home/End</b> 首/尾帧<br>
&nbsp;&nbsp;<b>-/+</b> 调整 FPS ±5<br><br>

<b>界面</b><br>
&nbsp;&nbsp;<b>Tab</b> 隐藏/显示侧栏 &nbsp; <b>H</b> 隐藏/显示 HUD<br>
&nbsp;&nbsp;<b>Ctrl+Enter</b> 演示模式 &nbsp; <b>F/F11</b> 全屏<br><br>

<b>其他</b><br>
&nbsp;&nbsp;<b>M</b> 截图 &nbsp; <b>Esc</b> 退出
""")

    # ═══════════════════════════════════════════════════════════════════════
    # Lifecycle
    # ═══════════════════════════════════════════════════════════════════════

    def closeEvent(self, event):
        self._render_timer.stop()
        self._move_timer.stop()
        if self.seq:
            self.seq.shutdown()
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════════
# 构建播放器对象 (不改动核心逻辑)
# ═══════════════════════════════════════════════════════════════════════════════
def _build_objects(args):
    ply_path       = None
    sequence_dir   = None
    seq_mgr        = None
    cameras        = []
    config         = {'sh_degree': 3, 'white_background': False}

    input_path = getattr(args, 'input', None) or getattr(args, 'ply_path', None)

    if input_path:
        if os.path.isdir(input_path):
            ply_files = sorted(f for f in os.listdir(input_path) if f.endswith('.ply'))
            if ply_files:
                print(f"检测到序列目录: {input_path}  共 {len(ply_files)} 帧")
                sequence_dir = input_path
                ply_path     = os.path.join(input_path, ply_files[0])
            else:
                nested_candidates = []
                for rel in ("per_frame_ply", "global_per_frame_ply"):
                    cand = os.path.join(input_path, rel)
                    if os.path.isdir(cand):
                        nested_candidates.append(cand)

                window_dirs = sorted(
                    d for d in os.listdir(input_path)
                    if d.startswith("window_") and os.path.isdir(os.path.join(input_path, d))
                )
                for wd in window_dirs:
                    cand = os.path.join(input_path, wd, "per_frame_ply")
                    if os.path.isdir(cand):
                        nested_candidates.append(cand)

                for cand in nested_candidates:
                    cand_ply_files = sorted(f for f in os.listdir(cand) if f.endswith('.ply'))
                    if cand_ply_files:
                        print(f"自动检测到序列目录: {cand}  共 {len(cand_ply_files)} 帧")
                        sequence_dir = cand
                        ply_path = os.path.join(cand, cand_ply_files[0])
                        break

                if ply_path is None and os.path.exists(os.path.join(input_path, "cfg_args")):
                    config = _core.parse_cfg_args(input_path)
                    pc_dir = os.path.join(input_path, "point_cloud")
                    iter_dir = _core.find_largest_iteration(pc_dir)
                    ply_path = os.path.join(pc_dir, iter_dir, "point_cloud.ply")
        elif input_path.endswith('.ply') and os.path.exists(input_path):
            ply_path = input_path
        else:
            print(f"错误: 无法识别的输入 '{input_path}'"); sys.exit(1)
    elif getattr(args, 'model_path', None):
        mp       = args.model_path
        config   = _core.parse_cfg_args(mp)
        pc_dir   = os.path.join(mp, "point_cloud")
        iter_dir = _core.find_largest_iteration(pc_dir)
        ply_path = os.path.join(pc_dir, iter_dir, "point_cloud.ply")
    else:
        print("错误: 请指定序列目录或 PLY 文件"); sys.exit(1)

    if ply_path is None:
        print(
            "错误: 在输入目录中未找到可播放的 PLY 帧。\n"
            "请传入包含 .ply 的目录，或直接传入 data/.../global_per_frame_ply。"
        )
        sys.exit(1)

    if not os.path.exists(ply_path):
        print(f"错误: 找不到 '{ply_path}'"); sys.exit(1)

    # 解析渲染分辨率
    render_w, render_h = 1080, 720
    if getattr(args, 'render_resolution', None):
        try:
            render_w, render_h = _core.parse_resolution(args.render_resolution)
        except ValueError as e:
            print(f"分辨率参数错误: {e}"); sys.exit(1)
    elif hasattr(args, 'width') and hasattr(args, 'height'):
        render_w, render_h = args.width, args.height

    sh_degree    = getattr(args, 'sh_degree', None) or config['sh_degree']
    playback_fps = getattr(args, 'playback_fps', 30.0) or 30.0

    # 加载相机
    data_path = getattr(args, 'path', None)
    sparse_path = getattr(args, 'sparse', None)

    if sparse_path and os.path.isdir(sparse_path):
        cameras = _core.load_cameras_from_colmap(sparse_path)
    elif data_path:
        cameras_json = os.path.join(data_path, "cameras.json")
        cameras      = _core.load_cameras_from_json(cameras_json)
    else:
        cameras = []

    # 序列管理器
    if sequence_dir:
        io_workers = _core.clamp_positive_int(getattr(args, 'io_workers', None), _recommended_io_workers())
        seq_mgr = SequenceManager(
            sequence_dir,
            sh_degree     = sh_degree,
            playback_fps  = playback_fps,
            load_mode     = getattr(args, 'load_mode', SequenceManager.LOAD_MODE_AUTO),
            gpu_cache_size= _core.clamp_positive_int(getattr(args, 'gpu_cache_size', 10), 10),
            cpu_cache_size= _core.clamp_positive_int(getattr(args, 'cpu_cache_size',  3),  3),
            prefetch_count= max(0, int(getattr(args, 'prefetch_count', 2))),
            io_workers    = io_workers,
            pin_memory    = not getattr(args, 'no_pin_memory', False),
            max_gaussians = getattr(args, 'max_gaussians', None),
        )

    bg_color = [1, 1, 1] if (getattr(args, 'white_background', False) or config['white_background']) else [0, 0, 0]

    # 点云
    if seq_mgr:
        frame_device = "cuda" if torch.cuda.is_available() else "cpu"
        frame = seq_mgr.get_current_frame_data(prefer_device=frame_device)
        pc    = GaussianPointCloud.from_frame(frame)
    else:
        pc = GaussianPointCloud(ply_path, sh_degree=sh_degree,
                                max_gaussians=getattr(args, 'max_gaussians', None))

    # 相机
    camera = InteractiveCamera(
        render_w, render_h,
        cameras_info  = cameras,
        scene_center  = pc.scene_center,
        scene_extent  = pc.scene_extent,
    )
    if cameras:
        camera.set_camera(0)

    renderer = GaussianRenderer(pc, bg_color=bg_color)
    renderer.set_render_mode(getattr(args, 'display_mode', GaussianRenderer.RENDER_MODE_SPLAT))
    renderer.set_point_style(
        size=getattr(args, 'point_size', 1.0),
        opacity=getattr(args, 'point_opacity', 1.0),
    )
    return renderer, camera, seq_mgr, render_w, render_h


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="4DGS Interactive Player (Qt UI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python player.py data/1/window_000/per_frame_ply
  python player.py data/1/window_000/per_frame_ply --render-resolution 4k --playback-fps 30
  python player.py data/1/window_000/per_frame_ply --gpu-cache-size 6 --cpu-cache-size 16
  python player.py model/point_cloud/iteration_30000/point_cloud.ply --render-resolution 2k
  python player.py data/sequence --sparse data/sparse
""",
    )
    parser.add_argument("input",              nargs="?", default=None,   help="PLY 文件或序列目录")
    parser.add_argument("--model-path","-m",  default=None, dest="model_path")
    parser.add_argument("--path","-s",        default=None,              help="cameras.json 所在目录")
    parser.add_argument("--sparse",           default=None,              help="COLMAP sparse 重建目录")
    parser.add_argument("--ply_path",         default=None)
    parser.add_argument("--render-resolution",default="1080p",           help="渲染分辨率: 720p/1080p/2k/4k/WxH")
    parser.add_argument("--playback-fps",     type=float, default=30.0,  help="播放帧率")
    parser.add_argument("--sh_degree",        type=int,   default=None)
    parser.add_argument("--load-mode",        default=SequenceManager.LOAD_MODE_AUTO,
                        choices=[SequenceManager.LOAD_MODE_AUTO,
                                 SequenceManager.LOAD_MODE_STREAM,
                                 SequenceManager.LOAD_MODE_PRELOAD_CPU,
                                 SequenceManager.LOAD_MODE_PRELOAD_GPU])
    parser.add_argument("--gpu-cache-size",   type=int, default=10)
    parser.add_argument("--cpu-cache-size",   type=int, default=30)
    parser.add_argument("--prefetch-count",   type=int, default=30)
    parser.add_argument("--io-workers",       type=int, default=24)
    parser.add_argument("--max-gaussians",    type=int, default=None)
    parser.add_argument("--no-pin-memory",    action="store_true")
    parser.add_argument("--white_background","-w", action="store_true")
    parser.add_argument(
        "--display-mode",
        default=GaussianRenderer.RENDER_MODE_SPLAT,
        choices=[mode for _label, mode in [
            ("splat", GaussianRenderer.RENDER_MODE_SPLAT),
            ("points", GaussianRenderer.RENDER_MODE_POINTS),
            ("ring", GaussianRenderer.RENDER_MODE_RING),
        ]],
        help="显示模式",
    )
    parser.add_argument("--point-size",    type=float, default=1.0)
    parser.add_argument("--point-opacity", type=float, default=1.0)

    args = parser.parse_args()

    if args.input is None and args.model_path is None:
        parser.print_help()
        print("\n错误: 请指定序列目录或 PLY 文件")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setApplicationName("4DGS Viewer")

    # Fusion 暗色主题
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(10,  17,  24))
    pal.setColor(QPalette.WindowText,      QColor(232, 240, 248))
    pal.setColor(QPalette.Base,            QColor(10,  17,  24))
    pal.setColor(QPalette.AlternateBase,   QColor(15,  25,  35))
    pal.setColor(QPalette.Text,            QColor(232, 240, 248))
    pal.setColor(QPalette.Button,          QColor(19,  31,  44))
    pal.setColor(QPalette.ButtonText,      QColor(232, 240, 248))
    pal.setColor(QPalette.Highlight,       QColor(43,  125, 233))
    pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)

    print("正在初始化渲染引擎，请稍候…")
    renderer, camera, seq_mgr, rw, rh = _build_objects(args)

    win = MainWindow(renderer, camera, seq_mgr, render_w=rw, render_h=rh)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
