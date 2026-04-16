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
    ("Depth",    "depth"),      # TODO: depth visualization
    ("Alpha",    "alpha"),      # TODO: alpha visualization
    ("Motion",   "motion"),     # TODO: motion visualization
    ("Layer ID", "layer_id"),   # TODO: layer ID visualization
]

# Quick-access visualization modes shown as viewport capsule buttons
QUICK_VIS_MODES = ["rgb", "gaussian", "depth", "motion"]

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

# ── Project modes ─────────────────────────────────────────────────────────────
PROJECT_MODES = ["3DGS", "4DGS", "Hybrid"]


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
        self.antialiasing: bool = False

        # ── Gaussian params ──
        self.point_size: float = 1.0
        self.splat_scale: float = 1.0
        self.alpha_scale: float = 1.0
        self.show_gaussian_centers: bool = False
        self.show_ellipsoids: bool = False
        self.show_point_cloud: bool = False
        self.show_motion_trails: bool = False

        # ── 4DGS Layers ──
        self.layer_static: bool = True
        self.layer_persistent: bool = True
        self.layer_ephemeral: bool = True
        self.layer_spawned: bool = True
        self.layer_pruned: bool = False
        self.show_camera_frustums: bool = False

        # ── Debug ──
        self.show_active_set: bool = False
        self.show_visible_gaussians: bool = False
        self.show_cache_region: bool = False
        self.show_window_interval: bool = False
        self.show_diagnostics: bool = False
        self.show_bounding_boxes: bool = False

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
        self.gpu_name: str = ""
        self.cpu_count: int = 0
        self.gpu_memory_used: str = "—"
        self.gpu_memory_total: str = "—"
        self.cache_status: str = ""
        self.load_mode: str = ""
        self.cache_hit_rate: float = 0.0
        self.cache_window: str = ""
