"""
ViewportOverlay — floating HUD elements on top of the render viewport.

Layout:
  ┌─────────────────────────────────────────────────┐
  │ [Scene / Mode / Camera]     [RGB][Depth][Motion] │
  │                              [Gaussian]          │
  │                                                   │
  │                                                   │
  │ [操作提示]                        FPS: 60         │
  │                                  Frame: 1/120     │
  │                                  Cache: 100%      │
  └─────────────────────────────────────────────────┘
"""

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
)
from PyQt5.QtCore import Qt, pyqtSignal

from .state import UIState, QUICK_VIS_MODES, VISUALIZATION_MODES
from .style import C_TEXT, C_TEXT2, C_ACCENT, C_BG, C_BORDER, F_CAPTION, F_BODY, S


class ViewportOverlay(QWidget):
    """Manages all floating overlay elements on the viewport."""

    quick_vis_mode_clicked = pyqtSignal(str)

    def __init__(self, parent_widget: QWidget, state: UIState):
        super().__init__(parent_widget)
        self.state = state
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

        # We don't use a layout for self — children are positioned absolutely
        self._margin = 12

        self._build_scene_label(parent_widget)
        self._build_mode_capsules(parent_widget)
        self._build_shortcuts_label(parent_widget)
        self._build_hud_label(parent_widget)

    # ── Build elements ────────────────────────────────────────────────────

    def _build_scene_label(self, parent):
        """Top-left: scene title, display mode, camera mode."""
        self.scene_label = QLabel(parent)
        self.scene_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.scene_label.setStyleSheet(
            f"background: rgba(10,17,24,0.75); color: {C_TEXT}; "
            f"border-radius: {S(8)}px; padding: {S(6)}px {S(10)}px; font-size: {F_CAPTION()}px;"
        )
        self.scene_label.setText("Scene | RGB | Free")
        self.scene_label.adjustSize()

    def _build_mode_capsules(self, parent):
        """Top-right: quick visualization mode switch capsule buttons."""
        self.capsule_container = QWidget(parent)
        self.capsule_container.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(self.capsule_container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._capsule_btns = {}
        vis_labels = {key: label for label, key in VISUALIZATION_MODES}

        for mode_key in QUICK_VIS_MODES:
            btn = QPushButton(vis_labels.get(mode_key, mode_key))
            btn.setObjectName("CapsuleButton")
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.clicked.connect(lambda _, k=mode_key: self._on_capsule_clicked(k))
            layout.addWidget(btn)
            self._capsule_btns[mode_key] = btn

        self.capsule_container.adjustSize()
        self._sync_capsules()

    def _build_shortcuts_label(self, parent):
        """Bottom-left: operation hints."""
        self.shortcuts_label = QLabel(parent)
        self.shortcuts_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.shortcuts_label.setStyleSheet(
            f"background: rgba(10,17,24,0.6); color: {C_TEXT2}; "
            f"border-radius: {S(8)}px; padding: {S(6)}px {S(10)}px; font-size: {F_CAPTION()}px;"
        )
        self.shortcuts_label.setText(
            "拖动: 旋转  |  右键/中键: 平移  |  滚轮: 缩放\n"
            "Space: 播放  |  G: 显示  |  Tab: 面板  |  H: HUD"
        )
        self.shortcuts_label.adjustSize()

    def _build_hud_label(self, parent):
        """Bottom-right: compact HUD stats."""
        self.hud_label = QLabel(parent)
        self.hud_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.hud_label.setStyleSheet(
            f"background: rgba(10,17,24,0.75); color: {C_TEXT}; "
            f"border-radius: {S(8)}px; padding: {S(6)}px {S(10)}px; font-size: {F_BODY()}px; "
            f"font-family: 'Consolas', 'SF Mono', monospace;"
        )
        self.hud_label.setText("FPS —  |  Frame —  |  Cache —")
        self.hud_label.adjustSize()

    # ── Positioning ───────────────────────────────────────────────────────

    def reposition(self, w: int, h: int):
        """Reposition all overlay elements for viewport size (w, h)."""
        m = self._margin

        # Top-left: scene label
        self.scene_label.adjustSize()
        self.scene_label.move(m, m)

        # Top-right: capsule buttons
        self.capsule_container.adjustSize()
        cw = self.capsule_container.sizeHint().width()
        self.capsule_container.move(w - cw - m, m)

        # Bottom-left: shortcuts
        self.shortcuts_label.adjustSize()
        self.shortcuts_label.move(m, h - self.shortcuts_label.height() - m)

        # Bottom-right: HUD
        self.hud_label.adjustSize()
        hw = self.hud_label.width()
        hh = self.hud_label.height()
        self.hud_label.move(w - hw - m, h - hh - m)

    # ── Update ────────────────────────────────────────────────────────────

    def update_scene_label(self, scene_name: str, vis_mode: str, camera_mode: str):
        vis_labels = {key: label for label, key in VISUALIZATION_MODES}
        vis_str = vis_labels.get(vis_mode, vis_mode)
        self.scene_label.setText(f"{scene_name}  |  {vis_str}  |  {camera_mode}")
        self.scene_label.adjustSize()

    def update_hud(self, fps: float, frame_str: str, cache_str: str):
        self.hud_label.setText(f"FPS {fps:.0f}  |  {frame_str}  |  {cache_str}")
        self.hud_label.adjustSize()

    def sync_vis_mode(self, mode_key: str):
        self.state.vis_mode = mode_key
        self._sync_capsules()

    def set_hud_visible(self, visible: bool):
        self.hud_label.setVisible(visible)
        self.scene_label.setVisible(visible)

    def set_shortcuts_visible(self, visible: bool):
        self.shortcuts_label.setVisible(visible)

    def set_all_visible(self, visible: bool):
        self.scene_label.setVisible(visible)
        self.capsule_container.setVisible(visible)
        self.shortcuts_label.setVisible(visible)
        self.hud_label.setVisible(visible)

    # ── Internal ──────────────────────────────────────────────────────────

    def _sync_capsules(self):
        for key, btn in self._capsule_btns.items():
            btn.blockSignals(True)
            btn.setChecked(key == self.state.vis_mode)
            btn.blockSignals(False)

    def _on_capsule_clicked(self, mode_key: str):
        self.quick_vis_mode_clicked.emit(mode_key)
        self._sync_capsules()
