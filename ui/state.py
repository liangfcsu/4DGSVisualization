"""
Centralized UI state management for 4DGS Viewer.
"""


# ── Resolution presets ────────────────────────────────────────────────────────
RESOLUTION_OPTIONS = [
    ("720p    1280×720",  1280,  720),
    ("1080p  1920×1080", 1920, 1080),
    ("2K     2560×1440", 2560, 1440),
    ("4K     3840×2160", 3840, 2160),
]

FPS_PRESETS = [10, 15, 24, 30, 60, 120]

# ── Visualization modes ──────────────────────────────────────────────────────
VISUALIZATION_MODES = [
    ("RGB",      "rgb"),        # → splat render mode
    ("Gaussian", "gaussian"),   # → points render mode
    ("Ring",     "ring"),       # → ring render mode
]

# Quick-access visualization modes shown as viewport capsule buttons
QUICK_VIS_MODES = ["rgb", "gaussian", "ring"]

# Mapping from visualization mode key → renderer render_mode string
VIS_TO_RENDER_MODE = {
    "rgb":      "splat",
    "gaussian": "points",
    "ring":     "ring",
}

# ── Camera modes ──────────────────────────────────────────────────────────────
CAMERA_MODES = [
    ("自由视角",   "free"),       # FPS mode
    ("Trackball", "trackball"),
    ("环绕相机",   "orbit"),
]

CAMERA_MODE_TO_INDEX = {
    "free":      0,   # InteractiveCamera.MODE_FPS
    "trackball": 1,   # InteractiveCamera.MODE_TRACKBALL
    "orbit":     2,   # InteractiveCamera.MODE_ORBIT
}


class UIState:
    """Centralized state for all UI panels."""

    def __init__(self):
        # ── Scene ──
        self.scene_name: str = "Scene"
        self.project_mode: str = "4DGS"

        # ── Visualization ──
        self.vis_mode: str = "rgb"

        # ── Camera ──
        self.camera_mode: str = "free"

        # ── Display params ──
        self.render_w: int = 1920
        self.render_h: int = 1080
        self.background_idx: int = 0        # 0=black, 1=white
        self.gamma: float = 1.0
        self.exposure: float = 1.0

        # ── Gaussian params ──
        self.point_size: float = 1.0
        self.alpha_scale: float = 1.0
        self.ring_size: float = 0.3
        self.selection_mode: bool = False

        # ── Panel visibility ──
        self.left_panel_visible: bool = True
        self.right_panel_visible: bool = True
        self.hud_visible: bool = True
        self.presentation_mode: bool = False

        # ── Playback (read by panels) ──
        self.current_frame: int = 0
        self.total_frames: int = 1
        self.is_playing: bool = False
        self.playback_fps: float = 30.0
        self.loop_enabled: bool = True

        # ── Runtime stats (updated by main window) ──
        self.viewer_fps: float = 0.0
        self.window_w: int = 0
        self.window_h: int = 0
        self.total_gaussians: int = 0
        self.visible_gaussians: int = 0
        self.selected_gaussians: int = 0
        self.hidden_gaussians: int = 0
        self.deleted_gaussians: int = 0
        self.gpu_name: str = ""
        self.cpu_count: int = 0
        self.gpu_memory_used: str = "—"
        self.gpu_memory_total: str = "—"
        self.cache_status: str = ""
        self.load_mode: str = ""
        self.cache_hit_rate: float = 0.0
