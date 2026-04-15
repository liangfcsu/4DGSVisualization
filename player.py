#!/usr/bin/env python3
"""
4DGS Interactive Player — Qt UI 交互式播放器

用法:
    python player.py /home/lf/algorithm/3dgs/3dgsalgotithm/CodeReading/gaussian-splatting/output/douyinhuaban/output01/global_per_frame_ply
    python player.py data/1/window_000/per_frame_ply --render-resolution 4k --playback-fps 30
    python player.py data/1/window_000/per_frame_ply --load-mode stream --gpu-cache-size 4
    python player.py point_cloud.ply --render-resolution 2k
    python player.py <sequence_dir> --sparse data/sparse  # 加载 COLMAP 真实相机位姿

快捷键:
    1/2/3/4      - 切换渲染分辨率 (720p/1080p/2K/4K)
    G            - 切换显示模式 (高斯 / 点)
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
    F11          - 全屏
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

# ─── 第一步：加载核心模块（会间接 import cv2，cv2 会污染 QT_QPA_PLATFORM_PLUGIN_PATH）
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

# ─── 第二步：cv2 已被导入并可能污染了插件路径，在 PyQt5 导入前修正 ────────────
# cv2-python 会在 import cv2 时把 QT_QPA_PLATFORM_PLUGIN_PATH 指向
# cv2/qt/plugins（旧版 Qt），导致 PyQt5 的 xcb 插件无法加载并崩溃。
# 必须在 cv2 import 完成之后、PyQt5 import 之前覆盖这个环境变量。
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

# ─── 第三步：正常导入 PyQt5 ──────────────────────────────────────────────────
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


# ─── 分辨率预设 ────────────────────────────────────────────────────────────────
RESOLUTION_OPTIONS = [
    ("720p    1280×720",  1280,  720),
    ("1080p  1920×1080", 1920, 1080),
    ("2K     2560×1440", 2560, 1440),
    ("4K     3840×2160", 3840, 2160),
]

FPS_PRESETS = [10, 15, 24, 30, 60, 120]
DISPLAY_MODE_OPTIONS = [
    ("高斯", GaussianRenderer.RENDER_MODE_SPLAT),
    ("点模式", GaussianRenderer.RENDER_MODE_POINTS),
]


def _recommended_io_workers() -> int:
    cpu_count = os.cpu_count() or 4
    return max(4, min(8, cpu_count))

CAMERA_MOTION_KEYS = {
    Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D,
    Qt.Key_Q, Qt.Key_E, Qt.Key_I, Qt.Key_K,
    Qt.Key_J, Qt.Key_L, Qt.Key_U, Qt.Key_O,
}

PLAYER_QSS = """
QMainWindow, QWidget#CentralShell {
    background: #09111a;
    color: #e7eef7;
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei";
}
QMenuBar {
    background: #0d1722;
    color: #e7eef7;
    border-bottom: 1px solid #223446;
    padding: 4px 10px;
}
QMenuBar::item {
    padding: 7px 12px;
    border-radius: 8px;
}
QMenuBar::item:selected {
    background: #172435;
}
QMenu {
    background: #101a25;
    color: #e7eef7;
    border: 1px solid #24384b;
    padding: 6px;
}
QMenu::item {
    padding: 7px 16px;
    border-radius: 8px;
}
QMenu::item:selected {
    background: #18334a;
}
QToolBar#TopBar {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0f1a26, stop:1 #142536);
    border: none;
    border-bottom: 1px solid #24384b;
    spacing: 8px;
    padding: 10px 12px;
}
QFrame#RenderShell {
    background: qradialgradient(cx:0.5, cy:0.2, radius:1.15, fx:0.5, fy:0.15, stop:0 #132132, stop:1 #09111a);
}
QFrame#ControlBar {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #101a25, stop:1 #162636);
    border-top: 1px solid #24384b;
}
QStatusBar#AppStatusBar {
    background: #0d1722;
    color: #9cb4c9;
    border-top: 1px solid #223446;
}
QLabel#SectionLabel {
    color: #87a1ba;
    font-size: 12px;
    font-weight: 600;
}
QLabel#FrameCounter {
    color: #f5f9fd;
    font-size: 18px;
    font-weight: 700;
}
QLabel#InfoLabel {
    color: #d7e3ef;
    background: #0d1620;
    border: 1px solid #273b4f;
    border-radius: 12px;
    padding: 6px 10px;
}
QLabel#ShortcutHint {
    color: #dce8f4;
    background: rgba(19, 35, 52, 0.95);
    border: 1px solid #31506e;
    border-radius: 12px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
}
QLabel#SceneBadge {
    color: #e9f4ff;
    background: #132236;
    border: 1px solid #365779;
    border-radius: 12px;
    padding: 6px 12px;
    font-weight: 600;
}
QPushButton {
    background: #172536;
    color: #eff7ff;
    border: 1px solid #2b435a;
    border-radius: 12px;
    padding: 7px 12px;
    min-height: 34px;
}
QPushButton:hover {
    background: #203247;
    border-color: #44637f;
}
QPushButton:pressed {
    background: #132233;
}
QPushButton#PrimaryButton {
    background: #1e6ee5;
    border: 1px solid #418cff;
}
QPushButton#PrimaryButton:hover {
    background: #3584ff;
}
QPushButton#TransportButton {
    min-width: 38px;
    min-height: 38px;
    padding: 0px;
    border-radius: 12px;
}
QPushButton#ChipButton {
    min-width: 38px;
    min-height: 28px;
    padding: 0 8px;
    border-radius: 10px;
    background: #111b26;
}
QPushButton#ChipButton:hover {
    background: #182736;
}
QPushButton:checked {
    background: #1e6ee5;
    border-color: #418cff;
}
QComboBox, QDoubleSpinBox {
    background: #101923;
    color: #f2f7fb;
    border: 1px solid #2b435a;
    border-radius: 12px;
    padding: 6px 10px;
    min-height: 34px;
    selection-background-color: #1e6ee5;
}
QComboBox:hover, QDoubleSpinBox:hover {
    border-color: #466884;
}
QComboBox::drop-down {
    border: none;
    width: 26px;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #18283a;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1e6ee5, stop:1 #49a5ff);
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #f3f8ff;
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
    border: 2px solid #1e6ee5;
}
QToolTip {
    background: #0d1620;
    color: #e7eef7;
    border: 1px solid #2a4055;
    padding: 6px 8px;
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# RenderView — 渲染显示控件（鼠标交互）
# ═══════════════════════════════════════════════════════════════════════════════
class RenderView(QWidget):
    """渲染区域控件：
    - 显示 GPU 渲染结果（自动缩放到控件大小，保持宽高比，黑边填充）
    - 捕获鼠标/键盘，驱动 InteractiveCamera
    """

    def __init__(self, parent=None):
        super().__init__(parent)
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
        self._interaction_callback = None

        self._left  = False
        self._right = False
        self._mid   = False
        self._last_pos = None
        self._trackball_ratio = 0.75

    def set_camera(self, camera: InteractiveCamera):
        self.camera = camera

    def set_interaction_callback(self, callback):
        self._interaction_callback = callback

    def is_interacting(self):
        return self._left or self._right or self._mid

    def update_image(self, rgb: np.ndarray):
        """接受 (H, W, 3) uint8 RGB 数组，缩放后绘制"""
        self._image_buffer = np.ascontiguousarray(rgb)
        h, w = self._image_buffer.shape[:2]
        qi = QImage(self._image_buffer.data, w, h, w * 3, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qi)
        self._scaled_pixmap = None
        self._scaled_size = None
        self._scaled_key = None
        self.update()  # 触发 paintEvent

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
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
            painter.drawPixmap(x, y, scaled)

    def resizeEvent(self, event):
        self._scaled_pixmap = None
        self._scaled_size = None
        self._scaled_key = None
        super().resizeEvent(event)

    def _in_center(self, pos):
        cx, cy = self.width() / 2, self.height() / 2
        r = min(self.width(), self.height()) / 2 * self._trackball_ratio
        return (pos.x() - cx) ** 2 + (pos.y() - cy) ** 2 < r * r

    def mousePressEvent(self, e):
        if not self.camera:
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
        b = e.button()
        if   b == Qt.LeftButton:   self._left  = False
        elif b == Qt.RightButton:  self._right = False
        elif b == Qt.MiddleButton: self._mid   = False
        if not (self._left or self._right or self._mid):
            self._last_pos = None
        if self._interaction_callback:
            self._interaction_callback()

    def mouseMoveEvent(self, e):
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
            if self._left:  self.camera.move_right(dx * 0.1)
            if self._right: self.camera.move_up(-dy * 0.1)
            if self._mid:   self.camera.move_forward(-dy * 0.1)

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
# MainWindow — 主窗口
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
        self.renderer   = renderer
        self.camera     = camera
        self.seq        = seq
        self.pc         = renderer.pc
        self.render_w   = render_w
        self.render_h   = render_h
        self._preferred_frame_device = "cuda" if torch.cuda.is_available() else "cpu"

        self._fps_avg      = 0.0
        self._last_t       = time.time()
        self._keys_held: set = set()
        self._seeking      = False
        self._needs_render = True
        self._shortcuts = []

        self.setWindowTitle(f"4DGS Interactive Player · {self._scene_name()}")
        self.setFocusPolicy(Qt.StrongFocus)

        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()
        self._connect()
        self._sync_resolution_combo()
        self._sync_display_mode_combo()
        self._install_shortcuts()
        self._apply_window_theme()
        self._apply_initial_window_size()

        # 渲染定时器
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._on_render)
        self._render_timer.start(8)          # ~120 Hz UI 轮询上限

        # 相机连续运动定时器
        self._move_timer = QTimer(self)
        self._move_timer.timeout.connect(self._process_held_keys)
        self._move_timer.start(16)

    # ─── UI 构建 ──────────────────────────────────────────────────────────────

    def _build_menu(self):
        mb = self.menuBar()

        # 文件
        fm = mb.addMenu("文件(&F)")
        a = QAction("截图 (&M)", self); a.setShortcut("M"); a.triggered.connect(self._screenshot)
        fm.addAction(a)
        fm.addSeparator()
        a = QAction("退出 (&Q)", self); a.setShortcut("Ctrl+Q"); a.triggered.connect(self.close)
        fm.addAction(a)

        # 视图
        vm = mb.addMenu("视图(&V)")
        for label, w, h in RESOLUTION_OPTIONS:
            a = QAction(f"渲染: {label}", self)
            a.triggered.connect(lambda _, w=w, h=h: self._set_render_res(w, h))
            vm.addAction(a)
        vm.addSeparator()
        a = QAction("全屏 (&F11)", self); a.setShortcut("F11"); a.triggered.connect(self._toggle_fullscreen)
        vm.addAction(a)

        # 帮助
        hm = mb.addMenu("帮助(&H)")
        a = QAction("快捷键说明", self); a.triggered.connect(self._show_help)
        hm.addAction(a)

    def _build_toolbar(self):
        tb = QToolBar("工具栏", self)
        tb.setObjectName("TopBar")
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        self.addToolBar(Qt.TopToolBarArea, tb)

        # 渲染分辨率
        tb.addWidget(self._lbl("  渲染分辨率:"))
        self.res_combo = QComboBox()
        for label, *_ in RESOLUTION_OPTIONS:
            self.res_combo.addItem(label)
        self.res_combo.setMinimumWidth(170)
        self.res_combo.setToolTip("渲染分辨率独立于窗口大小，窗口可自由拖动")
        tb.addWidget(self.res_combo)

        tb.addSeparator()

        # 背景色
        tb.addWidget(self._lbl(" 背景:"))
        self.bg_combo = QComboBox()
        self.bg_combo.addItems(["黑色", "白色"])
        self.bg_combo.setFixedWidth(60)
        tb.addWidget(self.bg_combo)

        tb.addSeparator()

        tb.addWidget(self._lbl(" 显示:"))
        self.display_combo = QComboBox()
        for label, _mode in DISPLAY_MODE_OPTIONS:
            self.display_combo.addItem(label)
        self.display_combo.setFixedWidth(96)
        self.display_combo.setToolTip("切换高斯 / 点模式 (G)")
        tb.addWidget(self.display_combo)

        tb.addWidget(self._lbl(" 点大小:"))
        self.point_size_spin = QDoubleSpinBox()
        self.point_size_spin.setRange(0.25, 4.0)
        self.point_size_spin.setSingleStep(0.25)
        self.point_size_spin.setDecimals(2)
        self.point_size_spin.setValue(getattr(self.renderer, "point_size", 1.0))
        self.point_size_spin.setFixedWidth(62)
        self.point_size_spin.setToolTip("点模式下的高斯尺寸倍率")
        tb.addWidget(self.point_size_spin)

        tb.addSeparator()

        # 相机模式
        tb.addWidget(self._lbl(" 相机模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["FPS", "Trackball", "Orbit"])
        self.mode_combo.setFixedWidth(90)
        tb.addWidget(self.mode_combo)

        tb.addSeparator()

        # 重置相机
        btn = QPushButton("↺ 重置")
        btn.setToolTip("重置相机到初始位置 (R)")
        btn.clicked.connect(self._reset_camera)
        tb.addWidget(btn)

        tb.addSeparator()

        # 截图
        btn2 = QPushButton("📷 截图")
        btn2.setObjectName("PrimaryButton")
        btn2.setToolTip("保存截图 (M)")
        btn2.clicked.connect(self._screenshot)
        tb.addWidget(btn2)

        tb.addSeparator()
        
        # 相机选择（如果有加载的相机）
        if self.camera.cameras_info and len(self.camera.cameras_info) > 0:
            tb.addWidget(self._lbl(" 相机位姿:"))
            self.camera_combo = QComboBox()
            self.camera_combo.addItem("自由视角")
            for i, cam in enumerate(self.camera.cameras_info):
                self.camera_combo.addItem(f"{i+1}. {cam['name']}")
            self.camera_combo.setMinimumWidth(120)
            self.camera_combo.setToolTip("选择相机位姿 (N下一个 / P跳转最近)")
            tb.addWidget(self.camera_combo)
            tb.addSeparator()
        else:
            self.camera_combo = None
        
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)

        scene_badge = QLabel(self._scene_name())
        scene_badge.setObjectName("SceneBadge")
        tb.addWidget(scene_badge)

    def _build_central(self):
        central = QWidget()
        central.setObjectName("CentralShell")
        self.setCentralWidget(central)
        vl = QVBoxLayout(central)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        # 渲染视图
        view_shell = QFrame()
        view_shell.setObjectName("RenderShell")
        view_layout = QVBoxLayout(view_shell)
        view_layout.setContentsMargins(18, 18, 18, 10)
        view_layout.setSpacing(10)
        hint_row = QHBoxLayout()
        hint_row.setContentsMargins(0, 0, 0, 0)
        hint_row.addStretch()
        hint = QLabel("Space 播放/暂停   G 显示模式   Esc 关闭   ←/→ 切帧")
        hint.setObjectName("ShortcutHint")
        hint_row.addWidget(hint)
        view_layout.addLayout(hint_row)
        self.view = RenderView()
        self.view.set_camera(self.camera)
        self.view.set_interaction_callback(self._request_render)
        view_layout.addWidget(self.view, stretch=1)
        vl.addWidget(view_shell, stretch=1)

        # 底部控制栏
        vl.addWidget(self._build_control_bar())

    def _build_control_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("ControlBar")
        bar.setFixedHeight(self.seq and 98 or 56)
        vl = QVBoxLayout(bar)
        vl.setContentsMargins(12, 6, 12, 6)
        vl.setSpacing(4)

        if self.seq:
            # 进度条行
            seek_row = QHBoxLayout()
            self.lbl_cur   = QLabel("1")
            self.lbl_cur.setObjectName("FrameCounter")
            self.lbl_cur.setFixedWidth(40)
            self.seek      = QSlider(Qt.Horizontal)
            self.seek.setRange(0, max(0, self.seq.num_frames - 1))
            self.seek.setToolTip("拖动跳转帧位置")
            self.lbl_total = QLabel(str(self.seq.num_frames))
            self.lbl_total.setObjectName("SectionLabel")
            self.lbl_total.setFixedWidth(50)
            self.lbl_total.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            seek_row.addWidget(self.lbl_cur)
            seek_row.addWidget(self.seek, stretch=1)
            seek_row.addWidget(self.lbl_total)
            vl.addLayout(seek_row)

        # 按钮行
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        _BTN_STYLE = ""

        if self.seq:
            self.btn_first = self._ctrl_btn("⏮", "第一帧 (Home)", _BTN_STYLE)
            self.btn_prev  = self._ctrl_btn("⏪", "上一帧 (←)",   _BTN_STYLE)
            self.btn_play  = self._ctrl_btn("▶", "播放/暂停 (空格)", _BTN_STYLE, checkable=True)
            self.btn_next  = self._ctrl_btn("⏩", "下一帧 (→)",   _BTN_STYLE)
            self.btn_last  = self._ctrl_btn("⏭", "最后一帧 (End)", _BTN_STYLE)

            btn_row.addStretch()
            for b in (self.btn_first, self.btn_prev, self.btn_play, self.btn_next, self.btn_last):
                btn_row.addWidget(b)
            btn_row.addSpacing(20)

            # FPS 控制
            btn_row.addWidget(self._lbl("播放 FPS:"))
            self.fps_spin = QDoubleSpinBox()
            self.fps_spin.setRange(0.1, 240.0)
            self.fps_spin.setValue(self.seq.fps)
            self.fps_spin.setSingleStep(1.0)
            self.fps_spin.setDecimals(1)
            self.fps_spin.setFixedWidth(70)
            self.fps_spin.setToolTip("设置播放速度 (帧/秒)")
            btn_row.addWidget(self.fps_spin)
            btn_row.addSpacing(12)

            # FPS 预设快速按钮
            btn_row.addWidget(self._lbl("预设:"))
            for fps_val in FPS_PRESETS:
                fb = QPushButton(str(fps_val))
                fb.setObjectName("ChipButton")
                fb.setFixedSize(36, 28)
                fb.setToolTip(f"设置 {fps_val} FPS")
                fb.clicked.connect(lambda _, v=fps_val: self._set_fps_preset(v))
                btn_row.addWidget(fb)

        btn_row.addStretch()

        # 状态信息
        self.info_lbl = QLabel()
        self.info_lbl.setObjectName("InfoLabel")
        btn_row.addWidget(self.info_lbl)

        vl.addLayout(btn_row)
        return bar

    def _build_statusbar(self):
        status_bar = QStatusBar()
        status_bar.setObjectName("AppStatusBar")
        self.setStatusBar(status_bar)

    def _scene_name(self) -> str:
        if self.seq:
            return os.path.basename(os.path.abspath(self.seq.sequence_dir))
        if getattr(self.pc, "ply_path", None):
            return os.path.basename(self.pc.ply_path)
        return "Scene"

    def _apply_window_theme(self):
        self.setStyleSheet(PLAYER_QSS)

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
            self._register_shortcut(Qt.Key_Home, lambda: (self.seq.set_frame(0), self._reload()))
            self._register_shortcut(Qt.Key_End, lambda: (self.seq.set_frame(self.seq.num_frames - 1), self._reload()))
            self._register_shortcut(Qt.Key_Minus, lambda: self._adj_fps(-5.0))
            self._register_shortcut(Qt.Key_Equal, lambda: self._adj_fps(+5.0))
            self._register_shortcut(Qt.Key_Plus, lambda: self._adj_fps(+5.0))
        self._register_shortcut(Qt.Key_Escape, self.close)
        self._register_shortcut(Qt.Key_F11, self._toggle_fullscreen)
        self._register_shortcut(Qt.Key_G, self._cycle_display_mode)
        self._register_shortcut(Qt.Key_M, self._screenshot)
        self._register_shortcut(Qt.Key_Y, self._shortcut_trackball)
        self._register_shortcut(Qt.Key_B, self._shortcut_orbit)
        self._register_shortcut(Qt.Key_1, lambda: self._shortcut_resolution(0))
        self._register_shortcut(Qt.Key_2, lambda: self._shortcut_resolution(1))
        self._register_shortcut(Qt.Key_3, lambda: self._shortcut_resolution(2))
        self._register_shortcut(Qt.Key_4, lambda: self._shortcut_resolution(3))
        self._register_shortcut(Qt.Key_N, self._next_camera)
        self._register_shortcut(Qt.Key_P, self._snap_to_nearest_camera)
        self._register_shortcut(Qt.Key_R, self._reset_camera)

    def _shortcut_trackball(self):
        self.camera.switch_mode(InteractiveCamera.MODE_TRACKBALL)
        self.mode_combo.setCurrentIndex(1)
        self._request_render()

    def _shortcut_orbit(self):
        self.camera.switch_mode(InteractiveCamera.MODE_ORBIT)
        self.mode_combo.setCurrentIndex(2)
        self._request_render()

    def _cycle_display_mode(self):
        mode = self.renderer.cycle_render_mode()
        for idx, (_label, option_mode) in enumerate(DISPLAY_MODE_OPTIONS):
            if option_mode == mode:
                self.display_combo.blockSignals(True)
                self.display_combo.setCurrentIndex(idx)
                self.display_combo.blockSignals(False)
                break
        self.statusBar().showMessage(f"显示模式: {self.renderer.get_render_mode_label()}", 2500)
        self._request_render()

    def _shortcut_resolution(self, idx: int):
        if 0 <= idx < len(RESOLUTION_OPTIONS):
            _, w, h = RESOLUTION_OPTIONS[idx]
            self._set_render_res(w, h)
            self.res_combo.blockSignals(True)
            self.res_combo.setCurrentIndex(idx)
            self.res_combo.blockSignals(False)

    def _apply_initial_window_size(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 860)
            return

        available = screen.availableGeometry()
        chrome_w = 72
        chrome_h = 238 if self.seq else 172
        max_content_w = max(640, int(available.width() * 0.84) - chrome_w)
        max_content_h = max(360, int(available.height() * 0.8) - chrome_h)
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

    # ─── 信号连接 ─────────────────────────────────────────────────────────────

    def _connect(self):
        self.res_combo.currentIndexChanged.connect(self._on_res_changed)
        self.bg_combo.currentIndexChanged.connect(self._on_bg_changed)
        self.display_combo.currentIndexChanged.connect(self._on_display_mode_changed)
        self.point_size_spin.valueChanged.connect(self._on_point_size_changed)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        
        if self.camera_combo:
            self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)

        if self.seq:
            self.btn_first.clicked.connect(lambda: (self.seq.set_frame(0), self._reload()))
            self.btn_prev.clicked.connect(lambda: (self.seq.prev_frame(), self._reload()))
            self.btn_play.toggled.connect(self._on_play_toggled)
            self.btn_next.clicked.connect(lambda: (self.seq.next_frame(), self._reload()))
            self.btn_last.clicked.connect(lambda: (self.seq.set_frame(self.seq.num_frames - 1), self._reload()))
            self.seek.sliderPressed.connect(self._on_seek_pressed)
            self.seek.sliderReleased.connect(self._on_seek_released)
            self.seek.sliderMoved.connect(self._on_seek_moved)
            self.fps_spin.valueChanged.connect(lambda v: (self.seq.set_fps(v), self._request_render()))

    # ─── 渲染主循环 ───────────────────────────────────────────────────────────

    def _request_render(self):
        self._needs_render = True

    def _on_render(self):
        try:
            frame_updated = False

            # 序列推进
            if self.seq:
                self.seq.service_prefetch()
                if self.seq.playing:
                    due = self.seq.consume_due_frames()
                    if due > 0:
                        direction = 1 if self.seq.play_direction >= 0 else -1
                        ready_raw_idx = None
                        for offset in range(due, 0, -1):
                            raw_candidate = self.seq.current_frame + direction * offset
                            candidate = raw_candidate % self.seq.num_frames
                            if self.seq.is_frame_ready(candidate, prefer_device=self._preferred_frame_device):
                                ready_raw_idx = raw_candidate
                                break

                        if ready_raw_idx is not None:
                            self.seq.set_frame(ready_raw_idx)
                            self._reload()
                            frame_updated = True
                        else:
                            target_idx = (self.seq.current_frame + direction * due) % self.seq.num_frames
                            self.seq.request_frame(target_idx, prefer_device=self._preferred_frame_device)

            if not (self._needs_render or frame_updated or self.view.is_interacting()):
                return

            # GPU 渲染
            rendered = self.renderer.render(self.camera)
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
            self._needs_render = False
            self._update_ui()

        except Exception as e:
            self.statusBar().showMessage(f"渲染错误: {e}", 2000)

    def _reload(self):
        if self.seq:
            frame = self.seq.get_current_frame_data(prefer_device=self._preferred_frame_device)
            self.pc.apply_frame(frame)
        self._request_render()

    def _update_ui(self):
        now  = time.time()
        dt   = max(now - self._last_t, 1e-6)
        self._last_t = now
        self._fps_avg = 0.1 * (1.0 / dt) + 0.9 * self._fps_avg

        r_str = f"{self.render_w}×{self.render_h}"
        w_str = f"{self.view.width()}×{self.view.height()}"
        parts = [f"渲染: {r_str}", f"窗口: {w_str}", f"Viewer FPS: {self._fps_avg:.1f}"]
        parts.append(f"显示: {self.renderer.get_render_mode_label()}")

        if self.seq:
            f = self.seq.current_frame + 1
            t = self.seq.num_frames
            state = "▶ 播放" if self.seq.playing else "⏸ 暂停"
            parts += [f"帧: {f}/{t}", state, f"Playback: {self.seq.fps:.0f}fps"]

            # 同步 UI 控件（避免循环信号）
            if not self._seeking:
                self.seek.blockSignals(True)
                self.seek.setValue(self.seq.current_frame)
                self.seek.blockSignals(False)
            self.lbl_cur.setText(str(f))

            self.btn_play.blockSignals(True)
            playing = self.seq.playing
            self.btn_play.setChecked(playing)
            self.btn_play.setText("⏸" if playing else "▶")
            self.btn_play.blockSignals(False)

            stats = self.seq.get_load_stats()
            if stats['total_accesses'] > 0:
                parts.append(f"缓存命中: {stats['hit_rate']:.0f}%")
            parts.append(self.seq.get_cache_status())

        self.info_lbl.setText("  |  ".join(parts))
        self.statusBar().showMessage(
            f"渲染: {r_str}  窗口: {w_str}  FPS: {self._fps_avg:.1f}"
        )

    # ─── 键盘处理 ─────────────────────────────────────────────────────────────

    def keyPressEvent(self, e):
        k = e.key()
        if k in CAMERA_MOTION_KEYS:
            self._keys_held.add(k)
            self._request_render()
            e.accept()
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() in CAMERA_MOTION_KEYS:
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

    # ─── 控件事件槽 ───────────────────────────────────────────────────────────

    def _on_res_changed(self, idx):
        if 0 <= idx < len(RESOLUTION_OPTIONS):
            _, w, h = RESOLUTION_OPTIONS[idx]
            self._set_render_res(w, h)

    def _set_render_res(self, w: int, h: int):
        self.render_w = w
        self.render_h = h
        self.camera.resize(w, h)
        # 同步 combo（不触发信号）
        for i, (_, rw, rh) in enumerate(RESOLUTION_OPTIONS):
            if rw == w and rh == h:
                self.res_combo.blockSignals(True)
                self.res_combo.setCurrentIndex(i)
                self.res_combo.blockSignals(False)
                break
        # 清空 means2D 缓存
        self.renderer._means2d_buffer = None
        self.statusBar().showMessage(f"渲染分辨率: {w}×{h}", 3000)
        self._request_render()

    def _sync_resolution_combo(self):
        for i, (_, w, h) in enumerate(RESOLUTION_OPTIONS):
            if w == self.render_w and h == self.render_h:
                self.res_combo.setCurrentIndex(i)
                return
        self.res_combo.setCurrentIndex(0)

    def _sync_display_mode_combo(self):
        current_mode = getattr(self.renderer, "render_mode", GaussianRenderer.RENDER_MODE_SPLAT)
        for idx, (_label, mode) in enumerate(DISPLAY_MODE_OPTIONS):
            if mode == current_mode:
                self.display_combo.blockSignals(True)
                self.display_combo.setCurrentIndex(idx)
                self.display_combo.blockSignals(False)
                return
        self.display_combo.setCurrentIndex(0)

    def _on_bg_changed(self, idx):
        color = [1.0, 1.0, 1.0] if idx else [0.0, 0.0, 0.0]
        self.renderer.set_background_color(color)
        self._request_render()

    def _on_display_mode_changed(self, idx):
        if 0 <= idx < len(DISPLAY_MODE_OPTIONS):
            _label, mode = DISPLAY_MODE_OPTIONS[idx]
            self.renderer.set_render_mode(mode)
            self.statusBar().showMessage(f"显示模式: {self.renderer.get_render_mode_label()}", 2500)
            self._request_render()

    def _on_point_size_changed(self, value):
        self.renderer.set_point_style(size=value)
        self._request_render()

    def _on_mode_changed(self, idx):
        modes = [InteractiveCamera.MODE_FPS, InteractiveCamera.MODE_TRACKBALL, InteractiveCamera.MODE_ORBIT]
        if 0 <= idx < len(modes):
            self.camera.mode = modes[idx]
            self._request_render()
    
    def _on_camera_changed(self, idx):
        """相机选择下拉框回调"""
        if idx == 0:
            # 自由视角 - 重置相机
            self.camera.current_camera_idx = -1
        else:
            # 切换到指定相机
            camera_idx = idx - 1
            if self.camera.set_camera(camera_idx):
                pass
        self._request_render()
    
    def _next_camera(self):
        """切换到下一个相机 (N键)"""
        if self.camera.cameras_info and len(self.camera.cameras_info) > 0:
            self.camera.next_camera()
            # 同步下拉框
            if self.camera_combo:
                self.camera_combo.blockSignals(True)
                self.camera_combo.setCurrentIndex(self.camera.current_camera_idx + 1)
                self.camera_combo.blockSignals(False)
            self._request_render()
    
    def _snap_to_nearest_camera(self):
        """跳转到最近的相机 (P键)"""
        if self.camera.cameras_info and len(self.camera.cameras_info) > 0:
            self.camera.snap_to_nearest_camera()
            # 同步下拉框
            if self.camera_combo:
                self.camera_combo.blockSignals(True)
                self.camera_combo.setCurrentIndex(self.camera.current_camera_idx + 1)
                self.camera_combo.blockSignals(False)
            self._request_render()

    def _on_play_toggled(self, checked: bool):
        if self.seq:
            self.seq.playing = checked
            self.seq.last_update_time = time.time()
            self._reload()

    def _toggle_play(self):
        if self.seq:
            self.seq.toggle_play()
            self.btn_play.blockSignals(True)
            self.btn_play.setChecked(self.seq.playing)
            self.btn_play.setText("⏸" if self.seq.playing else "▶")
            self.btn_play.blockSignals(False)
            self._reload()

    def _on_seek_pressed(self):
        self._seeking = True

    def _on_seek_moved(self, value: int):
        if not self.seq:
            return
        self.lbl_cur.setText(str(value + 1))
        self.seq.set_frame(value)
        self.seq.request_frame(value, prefer_device=self._preferred_frame_device)

    def _on_seek_released(self):
        self._seeking = False
        if self.seq:
            self.seq.set_frame(self.seek.value())
            self._reload()

    def _adj_fps(self, delta: float):
        if self.seq:
            new_fps = self.seq.adjust_fps(delta)
            self.fps_spin.blockSignals(True)
            self.fps_spin.setValue(new_fps)
            self.fps_spin.blockSignals(False)
            self._request_render()

    def _set_fps_preset(self, fps_val: int):
        if self.seq:
            self.seq.set_fps(fps_val)
            self.fps_spin.blockSignals(True)
            self.fps_spin.setValue(fps_val)
            self.fps_spin.blockSignals(False)
            self._request_render()

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
        self._request_render()

    def _reset_camera(self):
        self.camera.reset()
        self._request_render()

    def _screenshot(self):
        # 渲染到高分辨率（原始渲染大小）并保存
        rendered = self.renderer.render(self.camera)
        img = rendered.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8).copy()
        h, w = img.shape[:2]
        qi = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qi)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"screenshot_{ts}.png"
        pixmap.save(fname)
        self.statusBar().showMessage(f"截图已保存: {fname}  ({w}×{h})", 4000)
        print(f"截图已保存: {fname}  ({w}×{h})")

    def _show_help(self):
        QMessageBox.information(self, "快捷键帮助", """
<b>分辨率控制</b><br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>1/2/3/4</b> — 720p / 1080p / 2K / 4K<br>
&nbsp;&nbsp;&nbsp;&nbsp;工具栏下拉菜单 — 任意预设<br>
&nbsp;&nbsp;&nbsp;&nbsp;窗口边框可自由拖动，不影响渲染分辨率<br><br>

<b>显示模式</b><br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>G</b> — 高斯 / 点模式 循环切换<br>
&nbsp;&nbsp;&nbsp;&nbsp;工具栏显示下拉框 — 直接切换显示方式<br>
&nbsp;&nbsp;&nbsp;&nbsp;工具栏点大小 — 调整点模式下的高斯尺寸倍率<br><br>

<b>相机移动 (FPS 模式)</b><br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>W/S</b> — 前进 / 后退 &nbsp; <b>A/D</b> — 左移 / 右移 &nbsp; <b>Q/E</b> — 下 / 上<br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>I/K</b> — 俯仰 &nbsp; <b>J/L</b> — 偏航 &nbsp; <b>U/O</b> — 滚转<br>
&nbsp;&nbsp;&nbsp;&nbsp;鼠标左键拖动 — 横向平移<br>
&nbsp;&nbsp;&nbsp;&nbsp;鼠标右键拖动 — 纵向平移<br>
&nbsp;&nbsp;&nbsp;&nbsp;鼠标中键拖动 / 滚轮 — 缩放<br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>Y</b> — Trackball 模式 &nbsp; <b>B</b> — Orbit 模式<br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>R</b> — 重置相机到初始位置<br><br>

<b>真实相机位姿切换</b><br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>N</b> — 下一个相机位姿<br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>P</b> — 跳转到最近的相机位姿<br>
&nbsp;&nbsp;&nbsp;&nbsp;工具栏下拉菜单 — 直接选择特定相机<br>
&nbsp;&nbsp;&nbsp;&nbsp;使用 --sparse 参数加载 COLMAP 数据<br><br>

<b>序列播放</b><br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>空格</b> — 播放 / 暂停<br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>← →</b> — 上 / 下一帧 &nbsp; <b>Home/End</b> — 第一 / 最后一帧<br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>- +</b> — 调整播放 FPS ±5<br>
&nbsp;&nbsp;&nbsp;&nbsp;底部 FPS 数字框 / 预设按钮 — 直接设定<br><br>

<b>其他</b><br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>M</b> — 截图（保存原始渲染分辨率）<br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>F11</b> — 全屏<br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>Esc</b> — 退出
""")

    # ─── 生命周期 ─────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._render_timer.stop()
        self._move_timer.stop()
        if self.seq:
            self.seq.shutdown()
        event.accept()

    # ─── 工具函数 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _lbl(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("SectionLabel")
        return lbl

    @staticmethod
    def _ctrl_btn(text, tip, style, checkable=False) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("TransportButton")
        btn.setToolTip(tip)
        btn.setFixedSize(36, 36)
        if style:
            btn.setStyleSheet(style)
        if checkable:
            btn.setCheckable(True)
        return btn


# ═══════════════════════════════════════════════════════════════════════════════
# 构建播放器对象
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
                # 兼容上层目录输入：自动探测常见帧目录
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
        # 从 COLMAP sparse 数据加载
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
            gpu_cache_size= _core.clamp_positive_int(getattr(args, 'gpu_cache_size',  4), 4),
            cpu_cache_size= _core.clamp_positive_int(getattr(args, 'cpu_cache_size',  8), 8),
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
  # 序列播放（自动流式加载）
  python player.py data/1/window_000/per_frame_ply

  # 4K 渲染，30fps
  python player.py data/1/window_000/per_frame_ply --render-resolution 4k --playback-fps 30

  # 大序列，增大缓存
  python player.py data/1/window_000/per_frame_ply --gpu-cache-size 6 --cpu-cache-size 16

  # 单帧 PLY
  python player.py model/point_cloud/iteration_30000/point_cloud.ply --render-resolution 2k
  
  # 加载 COLMAP 真实相机位姿（可用 N/P 键切换）
  python player.py data/sequence --sparse data/sparse
  python player.py output/scene/point_cloud.ply --sparse colmap/sparse/0
""",
    )
    parser.add_argument("input",              nargs="?", default=None,   help="PLY 文件或序列目录")
    parser.add_argument("--model-path","-m",  default=None, dest="model_path")
    parser.add_argument("--path","-s",        default=None,              help="cameras.json 所在目录")
    parser.add_argument("--sparse",           default=None,              help="COLMAP sparse 重建目录 (包含 cameras.bin/txt 和 images.bin/txt)")
    parser.add_argument("--ply_path",         default=None)
    parser.add_argument("--render-resolution",default="1080p",           help="渲染分辨率: 720p/1080p/2k/4k/WxH  [默认: 1080p]")
    parser.add_argument("--playback-fps",     type=float, default=30.0,  help="播放帧率 [默认: 30]")
    parser.add_argument("--sh_degree",        type=int,   default=None)
    parser.add_argument("--load-mode",        default=SequenceManager.LOAD_MODE_AUTO,
                        choices=[SequenceManager.LOAD_MODE_AUTO,
                                 SequenceManager.LOAD_MODE_STREAM,
                                 SequenceManager.LOAD_MODE_PRELOAD_CPU,
                                 SequenceManager.LOAD_MODE_PRELOAD_GPU])
    parser.add_argument("--gpu-cache-size",   type=int, default=10, help="GPU 缓存帧数 (适合小规模场景，减少加载卡顿)")
    parser.add_argument("--cpu-cache-size",   type=int, default=30, help="CPU 缓存帧数 (适合大规模场景，减少加载卡顿)")
    parser.add_argument("--prefetch-count",   type=int, default=30, help="预取帧数 (适合大规模场景，减少加载卡顿)")
    parser.add_argument("--io-workers",       type=int, default=24, help="后台IO线程数 [默认: 自动 4-8]")
    parser.add_argument("--max-gaussians",    type=int, default=None,    help="每帧最多高斯点数（适合预览大规模场景）")
    parser.add_argument("--no-pin-memory",    action="store_true")
    parser.add_argument("--white_background","-w", action="store_true")
    parser.add_argument(
        "--display-mode",
        default=GaussianRenderer.RENDER_MODE_SPLAT,
        choices=[mode for _label, mode in DISPLAY_MODE_OPTIONS],
        help="显示模式: splat / points [默认: splat]",
    )
    parser.add_argument("--point-size", type=float, default=1.0, help="点模式下的高斯尺寸倍率 [默认: 1.0]")
    parser.add_argument("--point-opacity", type=float, default=1.0, help="点模式下的全局不透明度倍率 [默认: 1.0]")

    args = parser.parse_args()

    if args.input is None and args.model_path is None:
        parser.print_help()
        print("\n错误: 请指定序列目录或 PLY 文件")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setApplicationName("4DGS Interactive Player")

    # Fusion 暗色主题
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(40,  40,  40))
    pal.setColor(QPalette.WindowText,      QColor(220, 220, 220))
    pal.setColor(QPalette.Base,            QColor(25,  25,  25))
    pal.setColor(QPalette.AlternateBase,   QColor(50,  50,  50))
    pal.setColor(QPalette.Text,            QColor(220, 220, 220))
    pal.setColor(QPalette.Button,          QColor(55,  55,  55))
    pal.setColor(QPalette.ButtonText,      QColor(220, 220, 220))
    pal.setColor(QPalette.Highlight,       QColor(58,  123, 213))
    pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)

    print("正在初始化渲染引擎，请稍候…")
    renderer, camera, seq_mgr, rw, rh = _build_objects(args)

    win = MainWindow(renderer, camera, seq_mgr, render_w=rw, render_h=rh)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
