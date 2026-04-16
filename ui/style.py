"""
Global stylesheet and visual constants for 4DGS Viewer.
All sizes are DPI-aware via get_scale_factor().
"""

import os

# ── Color palette ─────────────────────────────────────────────────────────────
C_BG          = "#0a1118"
C_SURFACE     = "#0f1923"
C_CARD        = "#131f2c"
C_CARD_HOVER  = "#182a3c"
C_BORDER      = "#1e3044"
C_BORDER_L    = "#2a4460"
C_ACCENT      = "#2b7de9"
C_ACCENT_HVR  = "#3d8ef5"
C_TEXT         = "#e8f0f8"
C_TEXT2        = "#8fa8be"
C_TEXT3        = "#5c7a94"
C_SUCCESS     = "#2ecc71"
C_WARNING     = "#f39c12"

# ── DPI scale detection ──────────────────────────────────────────────────────

_cached_scale = None

def get_scale_factor() -> float:
    """Detect screen DPI and return a scale factor (1.0 = 96 DPI)."""
    global _cached_scale
    if _cached_scale is not None:
        return _cached_scale

    scale = 1.0
    # Check env overrides first
    env_scale = os.environ.get("QT_SCALE_FACTOR")
    if env_scale:
        try:
            scale = float(env_scale)
            _cached_scale = scale
            return scale
        except ValueError:
            pass

    try:
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            screen = app.primaryScreen()
            if screen:
                dpi = screen.logicalDotsPerInch()
                scale = max(1.0, dpi / 96.0)
                _cached_scale = scale
    except Exception:
        pass

    # Don't cache 1.0 fallback — allow re-detection when QApp is ready
    return scale


def S(base_value: int) -> int:
    """Scale a pixel value by DPI factor."""
    return max(1, int(base_value * get_scale_factor()))


# ── Spacing (scaled) ─────────────────────────────────────────────────────────
def SP_XS(): return S(4)
def SP_SM(): return S(8)
def SP_MD(): return S(12)
def SP_LG(): return S(16)
def SP_XL(): return S(24)

# ── Border radius ────────────────────────────────────────────────────────────
R_BTN   = 8
R_PANEL = 12
R_CARD  = 12

# ── Font sizes (base, will be scaled in QSS) ─────────────────────────────────
def F_TITLE():   return S(18)
def F_HEADING(): return S(15)
def F_BODY():    return S(14)
def F_CAPTION(): return S(12)

# ── Panel widths (scaled) ─────────────────────────────────────────────────────
def LEFT_PANEL_W():  return S(280)
def RIGHT_PANEL_W(): return S(300)


def build_qss() -> str:
    """Build the global QSS with current DPI scale factor."""
    s = get_scale_factor()
    # Scaled sizes
    f_title   = F_TITLE()
    f_heading = F_HEADING()
    f_body    = F_BODY()
    f_caption = F_CAPTION()
    r_btn     = S(R_BTN)
    r_card    = S(R_CARD)
    # Scaled widget sizes
    btn_h     = S(34)
    combo_h   = S(32)
    transport = S(40)
    chip_h    = S(28)
    capsule_h = S(30)
    slider_h  = S(6)
    handle_w  = S(16)
    chk_sz    = S(18)
    sb_w      = S(8)

    return f"""
/* ── Global ─────────────────────────────────────────────────────────── */
QMainWindow, QWidget#CentralWidget {{
    background: {C_BG};
    color: {C_TEXT};
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    font-size: {f_body}px;
}}

/* ── Menu bar ───────────────────────────────────────────────────────── */
QMenuBar {{
    background: {C_SURFACE};
    color: {C_TEXT};
    border-bottom: 1px solid {C_BORDER};
    padding: {S(3)}px {S(8)}px;
    font-size: {f_body}px;
}}
QMenuBar::item {{
    padding: {S(6)}px {S(10)}px;
    border-radius: {r_btn}px;
}}
QMenuBar::item:selected {{
    background: {C_CARD};
}}
QMenu {{
    background: {C_SURFACE};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    padding: {S(4)}px;
    border-radius: {r_btn}px;
    font-size: {f_body}px;
}}
QMenu::item {{
    padding: {S(6)}px {S(14)}px;
    border-radius: {S(6)}px;
}}
QMenu::item:selected {{
    background: {C_CARD_HOVER};
}}

/* ── Top Bar ────────────────────────────────────────────────────────── */
QToolBar#TopBar {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {C_SURFACE}, stop:1 #142536);
    border: none;
    border-bottom: 1px solid {C_BORDER};
    spacing: {S(6)}px;
    padding: {S(6)}px {S(10)}px;
    min-height: {S(44)}px;
    max-height: {S(52)}px;
}}

/* ── Side Panels ────────────────────────────────────────────────────── */
QWidget#LeftPanel, QWidget#RightPanel {{
    background: {C_SURFACE};
    border: none;
}}
QWidget#LeftPanel {{
    border-right: 1px solid {C_BORDER};
}}
QWidget#RightPanel {{
    border-left: 1px solid {C_BORDER};
}}

/* ── Bottom Bar ─────────────────────────────────────────────────────── */
QWidget#BottomBar {{
    background: {C_SURFACE};
    border-top: 1px solid {C_BORDER};
}}

/* ── Render View ────────────────────────────────────────────────────── */
QWidget#RenderView {{
    background: #000;
}}

/* ── Labels ─────────────────────────────────────────────────────────── */
QLabel {{
    color: {C_TEXT};
    background: transparent;
    font-size: {f_body}px;
}}
QLabel#SectionLabel {{
    color: {C_TEXT2};
    font-size: {f_caption}px;
    font-weight: 600;
}}
QLabel#PanelTitle {{
    color: {C_TEXT};
    font-size: {f_heading}px;
    font-weight: 700;
    padding: {S(4)}px 0px;
}}
QLabel#CardTitle {{
    color: {C_TEXT2};
    font-size: {f_caption}px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
QLabel#CardValue {{
    color: {C_TEXT};
    font-size: {f_body}px;
}}
QLabel#FrameCounter {{
    color: {C_TEXT};
    font-size: {f_heading}px;
    font-weight: 700;
}}
QLabel#AppTitle {{
    color: {C_TEXT};
    font-size: {f_title}px;
    font-weight: 700;
    padding: 0 {S(4)}px;
}}
QLabel#SceneBadge {{
    color: {C_TEXT};
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: {S(10)}px;
    padding: {S(4)}px {S(10)}px;
    font-size: {f_caption}px;
    font-weight: 600;
}}
QLabel#ModeBadge {{
    color: {C_ACCENT};
    background: rgba(43,125,233,0.12);
    border: 1px solid rgba(43,125,233,0.3);
    border-radius: {r_btn}px;
    padding: {S(3)}px {S(8)}px;
    font-size: {f_caption}px;
    font-weight: 600;
}}

/* ── Buttons ────────────────────────────────────────────────────────── */
QPushButton {{
    background: {C_CARD};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    border-radius: {r_btn}px;
    padding: {S(5)}px {S(10)}px;
    min-height: {btn_h}px;
    font-size: {f_body}px;
}}
QPushButton:hover {{
    background: {C_CARD_HOVER};
    border-color: {C_BORDER_L};
}}
QPushButton:pressed {{
    background: {C_SURFACE};
}}
QPushButton:checked {{
    background: {C_ACCENT};
    border-color: {C_ACCENT_HVR};
    color: #fff;
}}
QPushButton#PrimaryButton {{
    background: {C_ACCENT};
    border: 1px solid {C_ACCENT_HVR};
    color: #fff;
}}
QPushButton#PrimaryButton:hover {{
    background: {C_ACCENT_HVR};
}}
QPushButton#TransportButton {{
    min-width: {transport}px;
    min-height: {transport}px;
    max-width: {transport}px;
    max-height: {transport}px;
    padding: 0;
    border-radius: {S(10)}px;
    font-size: {f_body}px;
}}
QPushButton#ChipButton {{
    min-width: {chip_h}px;
    min-height: {chip_h}px;
    max-height: {chip_h}px;
    padding: 0 {S(6)}px;
    border-radius: {S(8)}px;
    background: {C_BG};
    font-size: {f_caption}px;
}}
QPushButton#ChipButton:hover {{
    background: {C_CARD};
}}
QPushButton#ChipButton:checked {{
    background: {C_ACCENT};
    border-color: {C_ACCENT_HVR};
    color: #fff;
}}
QPushButton#CapsuleButton {{
    min-height: {capsule_h}px;
    max-height: {capsule_h}px;
    padding: 0 {S(10)}px;
    border-radius: {capsule_h // 2}px;
    background: rgba(15,25,35,0.85);
    border: 1px solid {C_BORDER};
    font-size: {f_caption}px;
    font-weight: 600;
}}
QPushButton#CapsuleButton:hover {{
    background: rgba(24,42,60,0.9);
}}
QPushButton#CapsuleButton:checked {{
    background: rgba(43,125,233,0.85);
    border-color: {C_ACCENT_HVR};
    color: #fff;
}}
QPushButton#TogglePanelBtn {{
    min-width: {S(22)}px;
    max-width: {S(22)}px;
    min-height: {S(48)}px;
    border-radius: {S(4)}px;
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    padding: 0;
    font-size: {f_caption}px;
    color: {C_TEXT2};
}}
QPushButton#TogglePanelBtn:hover {{
    background: {C_CARD_HOVER};
}}
QPushButton#SectionToggle {{
    background: transparent;
    border: none;
    text-align: left;
    padding: {S(6)}px {S(8)}px;
    font-size: {f_body}px;
    font-weight: 600;
    color: {C_TEXT};
    border-radius: {S(6)}px;
}}
QPushButton#SectionToggle:hover {{
    background: {C_CARD};
}}

/* ── ComboBox / SpinBox ─────────────────────────────────────────────── */
QComboBox, QDoubleSpinBox, QSpinBox {{
    background: {C_BG};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    border-radius: {r_btn}px;
    padding: {S(4)}px {S(8)}px;
    min-height: {combo_h}px;
    font-size: {f_body}px;
    selection-background-color: {C_ACCENT};
}}
QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {{
    border-color: {C_BORDER_L};
}}
QComboBox::drop-down {{
    border: none;
    width: {S(24)}px;
}}
QComboBox QAbstractItemView {{
    background: {C_SURFACE};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    selection-background-color: {C_ACCENT};
    padding: {S(2)}px;
    font-size: {f_body}px;
}}

/* ── Slider ─────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: {slider_h}px;
    background: {C_CARD};
    border-radius: {slider_h // 2}px;
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {C_ACCENT}, stop:1 {C_ACCENT_HVR});
    border-radius: {slider_h // 2}px;
}}
QSlider::handle:horizontal {{
    background: {C_TEXT};
    width: {handle_w}px;
    margin: {-(handle_w // 2 - slider_h // 2)}px 0;
    border-radius: {handle_w // 2}px;
    border: 2px solid {C_ACCENT};
}}

/* ── CheckBox ───────────────────────────────────────────────────────── */
QCheckBox {{
    spacing: {S(6)}px;
    color: {C_TEXT};
    font-size: {f_body}px;
}}
QCheckBox::indicator {{
    width: {chk_sz}px;
    height: {chk_sz}px;
    border-radius: {S(4)}px;
    border: 1px solid {C_BORDER_L};
    background: {C_BG};
}}
QCheckBox::indicator:checked {{
    background: {C_ACCENT};
    border-color: {C_ACCENT_HVR};
}}
QCheckBox::indicator:hover {{
    border-color: {C_ACCENT};
}}

/* ── Scroll Area ────────────────────────────────────────────────────── */
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: {sb_w}px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {C_BORDER};
    border-radius: {sb_w // 2}px;
    min-height: {S(30)}px;
}}
QScrollBar::handle:vertical:hover {{
    background: {C_BORDER_L};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}

/* ── Tooltip ────────────────────────────────────────────────────────── */
QToolTip {{
    background: {C_SURFACE};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    padding: {S(5)}px {S(8)}px;
    border-radius: {S(6)}px;
    font-size: {f_caption}px;
}}

/* ── Info Card ──────────────────────────────────────────────────────── */
QFrame#InfoCard {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: {r_card}px;
    padding: 0px;
}}
"""


# Build once at import time — will be rebuilt if needed
GLOBAL_QSS = build_qss()
