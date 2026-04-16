"""
Reusable widget components for 4DGS Viewer UI.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QSizePolicy, QGridLayout,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor

from .style import C_CARD, C_BORDER, C_TEXT, C_TEXT2, C_ACCENT, SP_SM, SP_MD, R_CARD, F_CAPTION, S


class CollapsibleSection(QWidget):
    """A collapsible section with a toggle header and content area."""

    def __init__(self, title: str, parent=None, expanded=True):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._toggle = QPushButton(f"▾  {title}" if expanded else f"▸  {title}")
        self._toggle.setObjectName("SectionToggle")
        self._toggle.setCursor(Qt.PointingHandCursor)
        self._toggle.clicked.connect(self._on_toggle)
        layout.addWidget(self._toggle)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(SP_SM(), SP_SM(), SP_SM(), SP_SM())
        self._content_layout.setSpacing(SP_SM())
        self._content.setVisible(expanded)
        layout.addWidget(self._content)

        self._title = title
        self._expanded = expanded

    @property
    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def _on_toggle(self):
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        arrow = "▾" if self._expanded else "▸"
        self._toggle.setText(f"{arrow}  {self._title}")

    def set_expanded(self, expanded: bool):
        if self._expanded != expanded:
            self._on_toggle()


class InfoCard(QFrame):
    """A card-style info panel with title and key-value rows."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("InfoCard")
        self._rows = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SP_MD(), SP_SM(), SP_MD(), SP_SM())
        layout.setSpacing(2)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("CardTitle")
        layout.addWidget(title_lbl)

        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, S(4), 0, S(2))
        self._grid.setHorizontalSpacing(SP_SM())
        self._grid.setVerticalSpacing(S(2))
        self._grid.setColumnStretch(1, 1)
        layout.addLayout(self._grid)

        self._row_count = 0

    def add_row(self, key: str, default_value: str = "—") -> QLabel:
        """Add a key-value row. Returns the value QLabel for later updates."""
        key_lbl = QLabel(key)
        key_lbl.setStyleSheet(f"color: {C_TEXT2}; font-size: {F_CAPTION()}px;")
        val_lbl = QLabel(default_value)
        val_lbl.setObjectName("CardValue")
        val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self._grid.addWidget(key_lbl, self._row_count, 0)
        self._grid.addWidget(val_lbl, self._row_count, 1)
        self._rows[key] = val_lbl
        self._row_count += 1
        return val_lbl

    def set_value(self, key: str, value: str):
        if key in self._rows:
            self._rows[key].setText(value)


class ToastNotification(QLabel):
    """Lightweight toast notification overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            f"background: rgba(43,125,233,0.88); color: #fff; "
            f"border-radius: 8px; padding: 8px 18px; font-size: 13px; font-weight: 600;"
        )
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.hide()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_message(self, text: str, duration_ms: int = 2500):
        self.setText(text)
        self.adjustSize()
        self._reposition()
        self.show()
        self.raise_()
        self._timer.start(duration_ms)

    def _reposition(self):
        parent = self.parent()
        if parent:
            x = (parent.width() - self.width()) // 2
            y = parent.height() - self.height() - 60
            self.move(max(0, x), max(0, y))

    def reposition(self):
        if self.isVisible():
            self._reposition()


class Separator(QFrame):
    """Thin horizontal separator line."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setFixedHeight(1)
        self.setStyleSheet(f"background: {C_BORDER}; border: none;")
