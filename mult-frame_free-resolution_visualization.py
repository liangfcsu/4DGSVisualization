#!/usr/bin/env python3
"""
3D Gaussian Splatting Interactive Viewer
一个基于Python的交互式3DGS渲染查看器，功能类似于SIBR_viewers
支持与SIBR相同的命令行参数格式

使用方法:
    # 简化用法 - 直接指定PLY文件
   python mult-frame_free-resolution_visualization.py  data/1/window_000/per_frame_ply
    # 完整用法 (类似SIBR)
    python mult-frame_free-resolution_visualization.py \
        --model-path "USdata/1.27videoprocess/reundistorted" \
        --iteration 30000
    
    # 指定初始窗口大小 (可拖动调整)
    python mult-frame_free-resolution_visualization.py point_cloud.ply --width 1920 --height 1080

控制说明 (与SIBR一致):
=== FPS模式 (默认) ===
    W/A/S/D - 前进/左移/后退/右移
    Q/E - 下降/上升
    I/K - 向上/向下看 (Pitch)
    J/L - 向左/向右看 (Yaw)
    U/O - 左滚/右滚 (Roll)
    
=== Trackball模式 (按Y切换) ===
    鼠标左键(中心区) - 球面旋转
    鼠标左键(边缘) - 滚转旋转
    鼠标右键(中心区) - 平面平移
    鼠标右键(边缘) - Z轴前后移动
    鼠标滚轮 - 缩放

=== 模式切换 ===
    Y - 切换FPS/Trackball模式
    B - 切换FPS/Orbit模式

=== 相机控制 ===
    P - 跳转到最近相机视角
    N - 下一个相机
    
=== 其他 ===
    M - 截图保存
    ESC - 退出
"""

import os
import sys
import math
import torch
import numpy as np
import argparse
import json
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plyfile import PlyData
from numpy.lib import recfunctions as np_recfunctions
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

try:
    import pygame
    from pygame.locals import *
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("警告: pygame未安装，将使用OpenCV作为后备显示方案")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

if PYGAME_AVAILABLE:
    PYGAME_K_PLUS = getattr(pygame, "K_PLUS", pygame.K_EQUALS)
    PYGAME_K_KP_PLUS = getattr(pygame, "K_KP_PLUS", pygame.K_EQUALS)
    PYGAME_K_KP_MINUS = getattr(pygame, "K_KP_MINUS", pygame.K_MINUS)
else:
    PYGAME_K_PLUS = ord('+')
    PYGAME_K_KP_PLUS = ord('+')
    PYGAME_K_KP_MINUS = ord('-')


def find_largest_iteration(point_cloud_dir):
    """查找最大迭代次数的子目录"""
    if not os.path.exists(point_cloud_dir):
        return None
    
    pattern = re.compile(r'iteration_(\d+)')
    largest_num = -1
    largest_dir = None
    
    for entry in os.listdir(point_cloud_dir):
        match = pattern.match(entry)
        if match:
            num = int(match.group(1))
            if num > largest_num:
                largest_num = num
                largest_dir = entry
    
    return largest_dir


def parse_cfg_args(model_path):
    """解析cfg_args文件获取配置"""
    cfg_file = os.path.join(model_path, "cfg_args")
    config = {
        'sh_degree': 3,
        'white_background': False,
        'source_path': None
    }
    
    if os.path.exists(cfg_file):
        with open(cfg_file, 'r') as f:
            line = f.read()
            
        match = re.search(r'sh_degree=(\d+)', line)
        if match:
            config['sh_degree'] = int(match.group(1))
        
        if 'white_background=True' in line:
            config['white_background'] = True
        
        match = re.search(r"source_path='([^']+)'", line)
        if match:
            config['source_path'] = match.group(1)
    
    return config


def load_cameras_from_json(json_path):
    """从cameras.json加载相机参数"""
    cameras = []
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            cam_data = json.load(f)
        
        for cam in cam_data:
            cameras.append({
                'id': cam['id'],
                'name': cam['img_name'],
                'width': cam['width'],
                'height': cam['height'],
                'position': np.array(cam['position'], dtype=np.float32),
                'rotation': np.array(cam['rotation'], dtype=np.float32),
                'fx': cam['fx'],
                'fy': cam['fy']
            })
    
    return cameras


def focal2fov(focal, pixels):
    """将焦距转换为视场角"""
    return 2 * math.atan(pixels / (2 * focal))


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    """计算世界到视图的变换矩阵 (与原始3DGS一致)"""
    Rt = np.zeros((4, 4), dtype=np.float32)
    Rt[:3, :3] = R.T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return Rt.astype(np.float32)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    """计算投影矩阵 (与原始3DGS一致)"""
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    
    P = torch.zeros(4, 4, device="cuda")
    z_sign = 1.0
    
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    
    return P


RESOLUTION_PRESETS = {
    '720p': (1280, 720),
    '1080p': (1920, 1080),
    '2k': (2560, 1440),
    '1440p': (2560, 1440),
    '4k': (3840, 2160),
    '8k': (7680, 4320),
}

# 运行时可用的分辨率快捷键映射
RUNTIME_RESOLUTION_PRESETS = [
    ('720p', RESOLUTION_PRESETS['720p']),
    ('1080p', RESOLUTION_PRESETS['1080p']),
    ('2k', RESOLUTION_PRESETS['2k']),
    ('4k', RESOLUTION_PRESETS['4k']),
]

# 默认支持的播放FPS预设
PLAYBACK_FPS_PRESETS = [10, 15, 24, 30, 60, 120]

RUNTIME_RESOLUTION_KEYS = {
    pygame.K_1 if PYGAME_AVAILABLE else 49: ('720p', RESOLUTION_PRESETS['720p']),
    pygame.K_2 if PYGAME_AVAILABLE else 50: ('1080p', RESOLUTION_PRESETS['1080p']),
    pygame.K_3 if PYGAME_AVAILABLE else 51: ('2k', RESOLUTION_PRESETS['2k']),
    pygame.K_4 if PYGAME_AVAILABLE else 52: ('4k', RESOLUTION_PRESETS['4k']),
}


def compute_fovy_from_fovx(fovx, width, height):
    """在保持水平视场角不变的情况下，计算新的垂直视场角"""
    if width <= 0 or height <= 0:
        return fovx
    aspect = height / width
    return 2.0 * math.atan(math.tan(fovx * 0.5) * aspect)


def clamp_positive_int(value, fallback):
    if value is None:
        return fallback
    return max(1, int(value))


def parse_resolution(value, default_size=None, allow_auto=False):
    """解析分辨率字符串，支持 2k/4k/1920x1080/1920*1080/auto"""
    if value is None:
        return default_size
    
    text = value.strip().lower()
    if allow_auto and text == 'auto':
        return None
    
    if text in RESOLUTION_PRESETS:
        return RESOLUTION_PRESETS[text]
    
    match = re.fullmatch(r'(\d+)\s*[xX\*]\s*(\d+)', text)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    raise ValueError(
        f"无法解析分辨率 '{value}'，请使用 2k/4k/1080p/1280x720 这类格式"
    )


def fit_size_within_bounds(width, height, max_width, max_height, margin=0.92):
    """将窗口尺寸按比例限制在屏幕范围内"""
    width = max(1, int(width))
    height = max(1, int(height))
    max_width = max(1, int(max_width * margin))
    max_height = max(1, int(max_height * margin))
    
    if width <= max_width and height <= max_height:
        return width, height
    
    scale = min(max_width / width, max_height / height)
    return max(1, int(width * scale)), max(1, int(height * scale))


def build_sample_indices(total_points, max_gaussians):
    if max_gaussians is None or max_gaussians <= 0 or total_points <= max_gaussians:
        return None
    return np.linspace(0, total_points - 1, num=max_gaussians, dtype=np.int64)


def get_vertex_data_array(vertex):
    if isinstance(vertex, np.ndarray):
        data = vertex
    else:
        data = getattr(vertex, "data", vertex)
    if getattr(data, "dtype", None) is None or data.dtype.names is None:
        raise ValueError("PLY vertex 数据必须是带命名字段的结构化数组")
    return data


def read_structured_columns(data, field_names, sample_indices=None):
    if sample_indices is not None:
        data = data[sample_indices]

    if not field_names:
        row_count = len(data)
        return np.empty((row_count, 0), dtype=np.float32)

    if len(field_names) == 1:
        arr = np.asarray(data[field_names[0]], dtype=np.float32)
        return np.ascontiguousarray(arr.reshape(-1, 1))

    try:
        arr = np_recfunctions.structured_to_unstructured(
            data[list(field_names)],
            dtype=np.float32,
            copy=False,
        )
    except Exception:
        arr = np.stack(
            [np.asarray(data[name], dtype=np.float32) for name in field_names],
            axis=1,
        )

    return np.ascontiguousarray(arr)


@lru_cache(maxsize=32)
def resolve_gaussian_property_layout(property_names, requested_sh_degree):
    extra_f_names = tuple(
        sorted(
            (name for name in property_names if name.startswith("f_rest_")),
            key=lambda x: int(x.split('_')[-1]),
        )
    )
    scale_names = tuple(
        sorted(
            (name for name in property_names if name.startswith("scale_")),
            key=lambda x: int(x.split('_')[-1]),
        )
    )
    rot_names = tuple(
        sorted(
            (name for name in property_names if name.startswith("rot")),
            key=lambda x: int(x.split('_')[-1]),
        )
    )

    num_extra = len(extra_f_names)
    actual_sh_coeffs = num_extra // 3 + 1
    actual_sh_degree = int(math.sqrt(actual_sh_coeffs)) - 1 if actual_sh_coeffs > 0 else 0
    active_sh_degree = min(requested_sh_degree, actual_sh_degree)
    expected_extra = max(0, 3 * ((active_sh_degree + 1) ** 2 - 1))

    return (
        active_sh_degree,
        extra_f_names[:expected_extra],
        scale_names,
        rot_names,
    )


PLY_NUMPY_DTYPES = {
    'char': 'i1',
    'uchar': 'u1',
    'int8': 'i1',
    'uint8': 'u1',
    'short': '<i2',
    'ushort': '<u2',
    'int16': '<i2',
    'uint16': '<u2',
    'int': '<i4',
    'uint': '<u4',
    'int32': '<i4',
    'uint32': '<u4',
    'float': '<f4',
    'float32': '<f4',
    'double': '<f8',
    'float64': '<f8',
}

PLAYER_SEQUENCE_CACHE_VERSION = 1


def fast_load_ply_vertex_table(path):
    """快速读取 binary_little_endian PLY 的 vertex 表，失败时抛出异常让上层回退"""
    with open(path, 'rb') as f:
        first_line = f.readline().decode('ascii', errors='strict').strip()
        if first_line != 'ply':
            raise ValueError("不是合法的 PLY 文件")
        
        format_line = f.readline().decode('ascii', errors='strict').strip()
        if format_line != 'format binary_little_endian 1.0':
            raise ValueError(f"暂不支持的PLY格式: {format_line}")
        
        vertex_count = None
        dtype_fields = []
        in_vertex_element = False
        
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY header 不完整")
            line = line.decode('ascii', errors='strict').strip()
            if line == 'end_header':
                break
            if not line or line.startswith('comment'):
                continue
            
            parts = line.split()
            keyword = parts[0]
            if keyword == 'element':
                in_vertex_element = parts[1] == 'vertex'
                if in_vertex_element:
                    vertex_count = int(parts[2])
            elif keyword == 'property' and in_vertex_element:
                if parts[1] == 'list':
                    raise ValueError("暂不支持带 list property 的 PLY")
                prop_type = parts[1]
                prop_name = parts[2]
                if prop_type not in PLY_NUMPY_DTYPES:
                    raise ValueError(f"暂不支持的 property 类型: {prop_type}")
                dtype_fields.append((prop_name, PLY_NUMPY_DTYPES[prop_type]))
        
        if vertex_count is None or not dtype_fields:
            raise ValueError("PLY 中未找到 vertex 元素")
        
        header_size = f.tell()
    
    dtype = np.dtype(dtype_fields)
    return np.memmap(path, mode='r', dtype=dtype, offset=header_size, shape=(vertex_count,))


def make_sequence_cache_dir(sequence_dir, sh_degree, max_gaussians=None, cache_dir=None):
    if cache_dir:
        return os.path.abspath(cache_dir)

    sequence_abs = os.path.abspath(sequence_dir)
    variant = f"sh{int(sh_degree)}_max{max_gaussians if max_gaussians is not None else 'all'}"
    cache_root = os.path.join(sequence_abs, ".player_cache")
    return os.path.join(cache_root, variant)


def get_sequence_cache_frame_name(frame_file_name):
    base_name, _ = os.path.splitext(frame_file_name)
    return f"{base_name}.gscache"


def load_sequence_cache_metadata(cache_dir):
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_sequence_cache_compatible(meta, sequence_dir, frame_files, sh_degree, max_gaussians):
    if not meta:
        return False
    if int(meta.get("version", -1)) != PLAYER_SEQUENCE_CACHE_VERSION:
        return False
    if os.path.abspath(meta.get("source_dir", "")) != os.path.abspath(sequence_dir):
        return False
    if int(meta.get("sh_degree", -1)) != int(sh_degree):
        return False
    meta_max_gaussians = meta.get("max_gaussians")
    if meta_max_gaussians is not None:
        meta_max_gaussians = int(meta_max_gaussians)
    if meta_max_gaussians != (None if max_gaussians is None else int(max_gaussians)):
        return False
    if meta.get("frame_files") != list(frame_files):
        return False
    frame_cache_files = meta.get("frame_cache_files")
    if not isinstance(frame_cache_files, list) or len(frame_cache_files) != len(frame_files):
        return False
    return True


def build_sequence_cache(
    sequence_dir,
    sh_degree=3,
    max_gaussians=None,
    cache_dir=None,
    overwrite=False,
    verbose=True,
):
    frame_files = sorted([f for f in os.listdir(sequence_dir) if f.endswith('.ply')])
    if not frame_files:
        raise ValueError(f"在 {sequence_dir} 中未找到PLY文件")

    resolved_cache_dir = make_sequence_cache_dir(sequence_dir, sh_degree, max_gaussians, cache_dir=cache_dir)
    os.makedirs(resolved_cache_dir, exist_ok=True)

    meta = load_sequence_cache_metadata(resolved_cache_dir)
    if not overwrite and is_sequence_cache_compatible(meta, sequence_dir, frame_files, sh_degree, max_gaussians):
        frame_cache_paths = [os.path.join(resolved_cache_dir, name) for name in meta["frame_cache_files"]]
        if all(os.path.exists(path) for path in frame_cache_paths):
            if verbose:
                print(f"已存在可用播放器缓存: {resolved_cache_dir}")
            return resolved_cache_dir

    frame_cache_files = [get_sequence_cache_frame_name(name) for name in frame_files]
    total_frames = len(frame_files)
    start_time = time.time()

    if verbose:
        print(f"开始构建播放器缓存: {resolved_cache_dir}")
        print(f"  帧数: {total_frames}, SH度数: {sh_degree}, max_gaussians: {max_gaussians}")

    for idx, frame_file in enumerate(frame_files):
        src_path = os.path.join(sequence_dir, frame_file)
        cache_path = os.path.join(resolved_cache_dir, frame_cache_files[idx])
        if not overwrite and os.path.exists(cache_path):
            if verbose and (idx == total_frames - 1 or (idx + 1) % max(1, min(20, total_frames)) == 0):
                print(f"  缓存进度: {idx + 1}/{total_frames} (复用已有文件)")
            continue

        frame = GaussianFrame.from_ply(
            src_path,
            sh_degree=sh_degree,
            pin_memory=False,
            max_gaussians=max_gaussians,
            verbose=verbose and idx == 0,
        )
        frame.save_cache(cache_path)
        if verbose and (idx == total_frames - 1 or (idx + 1) % max(1, min(20, total_frames)) == 0):
            print(f"  缓存进度: {idx + 1}/{total_frames}")

    meta_payload = {
        "version": PLAYER_SEQUENCE_CACHE_VERSION,
        "source_dir": os.path.abspath(sequence_dir),
        "frame_files": frame_files,
        "frame_cache_files": frame_cache_files,
        "num_frames": total_frames,
        "sh_degree": int(sh_degree),
        "max_gaussians": None if max_gaussians is None else int(max_gaussians),
        "created_at": datetime.now().isoformat(),
    }
    meta_tmp_path = os.path.join(resolved_cache_dir, "meta.json.tmp")
    with open(meta_tmp_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, ensure_ascii=False, indent=2)
    os.replace(meta_tmp_path, os.path.join(resolved_cache_dir, "meta.json"))

    if verbose:
        print(f"播放器缓存构建完成，用时 {time.time() - start_time:.1f}s")
    return resolved_cache_dir


class GaussianFrame:
    """单帧高斯数据，可驻留在 CPU 或 GPU"""
    
    def __init__(
        self,
        xyz,
        features,
        scaling,
        rotation,
        opacity,
        sh_degree,
        scene_center,
        scene_extent,
        source_path,
        point_count,
        sampled_count,
    ):
        self.xyz = xyz
        self.features = features
        self.scaling = scaling
        self.rotation = rotation
        self.opacity = opacity
        self.sh_degree = sh_degree
        self.scene_center = scene_center.astype(np.float32)
        self.scene_extent = float(scene_extent)
        self.source_path = source_path
        self.point_count = int(point_count)
        self.sampled_count = int(sampled_count)
    
    @property
    def is_pinned(self):
        return self.xyz.device.type == 'cpu' and self.xyz.is_pinned()
    
    @property
    def tensor_count(self):
        return self.xyz.shape[0]
    
    @classmethod
    def _maybe_pin(cls, tensor, pin_memory):
        if not pin_memory or tensor.device.type != 'cpu':
            return tensor
        try:
            return tensor.pin_memory()
        except RuntimeError:
            return tensor
    
    @classmethod
    def from_ply(cls, path, sh_degree=3, pin_memory=False, max_gaussians=None, verbose=True):
        """从 PLY 读取单帧高斯数据，默认保留在 CPU 侧"""
        if verbose:
            print(f"正在加载点云: {path}")
        try:
            vertex = fast_load_ply_vertex_table(path)
            fast_loader_used = True
        except Exception:
            plydata = PlyData.read(path)
            vertex = plydata.elements[0]
            fast_loader_used = False

        data = get_vertex_data_array(vertex)
        total_points = len(data)
        sample_indices = build_sample_indices(total_points, max_gaussians)

        property_names = tuple(data.dtype.names)
        active_sh_degree, extra_f_names, scale_names, rot_names = resolve_gaussian_property_layout(
            property_names,
            sh_degree,
        )

        xyz = read_structured_columns(data, ("x", "y", "z"), sample_indices)
        opacities = read_structured_columns(data, ("opacity",), sample_indices)

        features_dc = read_structured_columns(
            data,
            ("f_dc_0", "f_dc_1", "f_dc_2"),
            sample_indices,
        )[:, np.newaxis, :]

        extra_coeff_count = max(0, (active_sh_degree + 1) ** 2 - 1)
        if extra_f_names and extra_coeff_count > 0:
            features_extra = read_structured_columns(data, extra_f_names, sample_indices)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, extra_coeff_count))
        else:
            features_extra = np.zeros((xyz.shape[0], 3, 0), dtype=np.float32)

        features = np.concatenate(
            (
                features_dc,
                np.transpose(features_extra, (0, 2, 1)),
            ),
            axis=1,
        )

        scales = read_structured_columns(data, scale_names, sample_indices)
        rots = read_structured_columns(data, rot_names, sample_indices)

        xyz_t = cls._maybe_pin(torch.from_numpy(xyz), pin_memory)
        features_t = cls._maybe_pin(torch.from_numpy(features).contiguous(), pin_memory)
        opacity_t = cls._maybe_pin(torch.from_numpy(1.0 / (1.0 + np.exp(-opacities))).contiguous(), pin_memory)
        scaling_t = cls._maybe_pin(torch.from_numpy(np.exp(scales)).contiguous(), pin_memory)
        if rots.shape[1] > 0:
            rot_norm = np.linalg.norm(rots, axis=1, keepdims=True)
            rot_norm = np.maximum(rot_norm, 1e-8)
            rots = rots / rot_norm
        rotation_t = cls._maybe_pin(torch.from_numpy(rots).contiguous(), pin_memory)
        
        scene_center = xyz.mean(axis=0).astype(np.float32)
        scene_min = xyz.min(axis=0)
        scene_max = xyz.max(axis=0)
        scene_extent = np.linalg.norm(scene_max - scene_min)
        
        sampled_count = xyz.shape[0]
        loader_name = "fast-memmap" if fast_loader_used else "plyfile"
        if verbose:
            if sample_indices is not None:
                print(
                    f"加载完成: 从 {total_points} 个高斯中采样 {sampled_count} 个, "
                    f"SH度数: {active_sh_degree}, 读取器: {loader_name}"
                )
            else:
                print(f"加载完成: {sampled_count} 个高斯点, SH度数: {active_sh_degree}, 读取器: {loader_name}")
        
        return cls(
            xyz=xyz_t,
            features=features_t,
            scaling=scaling_t,
            rotation=rotation_t,
            opacity=opacity_t,
            sh_degree=active_sh_degree,
            scene_center=scene_center,
            scene_extent=scene_extent,
            source_path=path,
            point_count=total_points,
            sampled_count=sampled_count,
        )
    
    def to_device(self, device, non_blocking=False):
        if self.xyz.device.type == device:
            return self
        return GaussianFrame(
            xyz=self.xyz.to(device=device, dtype=torch.float32, non_blocking=non_blocking),
            features=self.features.to(device=device, dtype=torch.float32, non_blocking=non_blocking),
            scaling=self.scaling.to(device=device, dtype=torch.float32, non_blocking=non_blocking),
            rotation=self.rotation.to(device=device, dtype=torch.float32, non_blocking=non_blocking),
            opacity=self.opacity.to(device=device, dtype=torch.float32, non_blocking=non_blocking),
            sh_degree=self.sh_degree,
            scene_center=self.scene_center.copy(),
            scene_extent=self.scene_extent,
            source_path=self.source_path,
            point_count=self.point_count,
            sampled_count=self.sampled_count,
        )

    def to_cache_payload(self):
        return {
            "version": PLAYER_SEQUENCE_CACHE_VERSION,
            "xyz": self.xyz.detach().cpu().contiguous(),
            "features": self.features.detach().cpu().contiguous(),
            "scaling": self.scaling.detach().cpu().contiguous(),
            "rotation": self.rotation.detach().cpu().contiguous(),
            "opacity": self.opacity.detach().cpu().contiguous(),
            "sh_degree": int(self.sh_degree),
            "scene_center": self.scene_center.copy(),
            "scene_extent": float(self.scene_extent),
            "source_path": self.source_path,
            "point_count": int(self.point_count),
            "sampled_count": int(self.sampled_count),
        }

    def save_cache(self, path):
        torch.save(self.to_cache_payload(), path)

    @classmethod
    def from_cache(cls, path, pin_memory=False, verbose=False):
        if verbose:
            print(f"正在加载缓存帧: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")

        def _cpu_tensor(value):
            if isinstance(value, torch.Tensor):
                tensor = value.detach().to(dtype=torch.float32, device="cpu").contiguous()
            else:
                tensor = torch.as_tensor(value, dtype=torch.float32).contiguous()
            return cls._maybe_pin(tensor, pin_memory)

        scene_center = payload.get("scene_center")
        if isinstance(scene_center, torch.Tensor):
            scene_center = scene_center.detach().cpu().numpy()
        scene_center = np.asarray(scene_center, dtype=np.float32)

        return cls(
            xyz=_cpu_tensor(payload["xyz"]),
            features=_cpu_tensor(payload["features"]),
            scaling=_cpu_tensor(payload["scaling"]),
            rotation=_cpu_tensor(payload["rotation"]),
            opacity=_cpu_tensor(payload["opacity"]),
            sh_degree=int(payload["sh_degree"]),
            scene_center=scene_center,
            scene_extent=float(payload["scene_extent"]),
            source_path=payload.get("source_path", path),
            point_count=int(payload.get("point_count", payload["xyz"].shape[0])),
            sampled_count=int(payload.get("sampled_count", payload["xyz"].shape[0])),
        )


class SequenceManager:
    """管理序列点云帧，默认使用流式 CPU/GPU 双缓存而非全量显存预加载"""
    
    LOAD_MODE_AUTO = "auto"
    LOAD_MODE_STREAM = "stream"
    LOAD_MODE_PRELOAD_CPU = "preload_cpu"
    LOAD_MODE_PRELOAD_GPU = "preload_gpu"
    
    def __init__(
        self,
        sequence_dir,
        sh_degree=3,
        playback_fps=30.0,
        load_mode=LOAD_MODE_STREAM,
        gpu_cache_size=2,
        cpu_cache_size=2,
        prefetch_count=1,
        pin_memory=True,
        max_gaussians=None,
        dynamic_cache=True,  # 动态调整缓存大小
        io_workers=2,
        max_pending_cpu=None,
        backward_prefetch=1,
        adaptive_prefetch=True,
        verbose_frame_loads=False,
        cache_dir=None,
        prefer_cache=True,
        cache_overwrite=False,
    ):
        self.sequence_dir = sequence_dir
        self.frame_files = sorted([f for f in os.listdir(sequence_dir) if f.endswith('.ply')])
        self.num_frames = len(self.frame_files)
        self.dynamic_cache = dynamic_cache
        self.load_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'load_time_avg': 0.0,
            'cpu_load_time_avg': 0.0,
            'gpu_upload_time_avg': 0.0,
        }
        self.frame_paths = [os.path.join(sequence_dir, file_name) for file_name in self.frame_files]
        self.total_sequence_bytes = sum(os.path.getsize(path) for path in self.frame_paths)
        self.current_frame = 0
        self.playing = False
        self.play_direction = 1
        self.fps = max(0.1, float(playback_fps))
        self.last_update_time = time.time()
        self.sh_degree = sh_degree
        self.load_mode = load_mode
        self.gpu_cache_size = max(1, int(gpu_cache_size))
        self.cpu_cache_size = max(1, int(cpu_cache_size))
        self.prefetch_count = max(0, int(prefetch_count))
        self.pin_memory = bool(pin_memory and torch.cuda.is_available())
        self.max_gaussians = max_gaussians
        self.io_workers = max(1, int(io_workers))
        if max_pending_cpu is None:
            max_pending_cpu = max(self.prefetch_count + 1, self.io_workers * 2)
        self.max_pending_cpu = max(1, int(max_pending_cpu))
        self.backward_prefetch = max(0, int(backward_prefetch))
        self.adaptive_prefetch = bool(adaptive_prefetch)
        self.verbose_frame_loads = bool(verbose_frame_loads)
        self.cache_dir = None
        self.cache_meta = None
        self.cache_frame_paths = None
        self.load_backend = "ply"
        
        self.cpu_cache = OrderedDict()
        self.gpu_cache = OrderedDict()
        self.pending_cpu = {}
        self.pending_gpu = OrderedDict()
        self.executor = None
        self.prefetch_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        self.scene_center = np.zeros(3, dtype=np.float32)
        self.scene_extent = 1.0
        self.active_sh_degree = sh_degree
        self.current_point_count = 0
        self.sampled_point_count = 0
        
        if self.num_frames == 0:
            raise ValueError(f"在 {sequence_dir} 中未找到PLY文件")

        if prefer_cache:
            resolved_cache_dir = make_sequence_cache_dir(
                sequence_dir,
                sh_degree,
                max_gaussians=max_gaussians,
                cache_dir=cache_dir,
            )
            meta = load_sequence_cache_metadata(resolved_cache_dir)
            has_compatible_cache = False
            if is_sequence_cache_compatible(meta, sequence_dir, self.frame_files, sh_degree, max_gaussians):
                frame_cache_paths = [os.path.join(resolved_cache_dir, name) for name in meta["frame_cache_files"]]
                has_compatible_cache = all(os.path.exists(path) for path in frame_cache_paths)
                if has_compatible_cache:
                    self.cache_dir = resolved_cache_dir
                    self.cache_meta = meta
                    self.cache_frame_paths = frame_cache_paths
                    self.load_backend = "cache"
                    print(f"检测到播放器缓存: {self.cache_dir}")

            if not has_compatible_cache:
                print(f"未找到可用播放器缓存，开始在序列目录内构建: {resolved_cache_dir}")
                build_sequence_cache(
                    sequence_dir,
                    sh_degree=sh_degree,
                    max_gaussians=max_gaussians,
                    cache_dir=resolved_cache_dir,
                    overwrite=cache_overwrite,
                    verbose=True,
                )
                meta = load_sequence_cache_metadata(resolved_cache_dir)
                if is_sequence_cache_compatible(meta, sequence_dir, self.frame_files, sh_degree, max_gaussians):
                    frame_cache_paths = [os.path.join(resolved_cache_dir, name) for name in meta["frame_cache_files"]]
                    if all(os.path.exists(path) for path in frame_cache_paths):
                        self.cache_dir = resolved_cache_dir
                        self.cache_meta = meta
                        self.cache_frame_paths = frame_cache_paths
                        self.load_backend = "cache"
                        print(f"播放器缓存已就绪: {self.cache_dir}")
        
        self.load_mode = self._resolve_load_mode(load_mode)
        print(f"找到 {self.num_frames} 帧序列")
        print(f"序列磁盘占用: {self.total_sequence_bytes / (1024 ** 3):.2f} GB")
        
        if self.load_mode == self.LOAD_MODE_PRELOAD_GPU:
            self._preload_all_frames_to_gpu()
        elif self.load_mode == self.LOAD_MODE_PRELOAD_CPU:
            self._preload_all_frames_to_cpu()
        else:
            self.executor = ThreadPoolExecutor(
                max_workers=self.io_workers,
                thread_name_prefix="ply-stream-loader",
            )
            print("使用流式加载模式: 小显存缓存 + CPU预取 + 后台读盘")
            print(
                f"  目标播放FPS: {self.fps:.1f}, GPU缓存: {self.gpu_cache_size}, "
                f"CPU缓存: {self.cpu_cache_size}, 预取帧数: {self.prefetch_count}, "
                f"IO线程: {self.io_workers}, 数据源: {self.load_backend}"
            )
            if self.max_gaussians:
                print(f"  每帧最大高斯数: {self.max_gaussians}")
            first_frame = self._load_frame_cpu_sync(0, verbose=True)
            self.scene_center = first_frame.scene_center.copy()
            self.scene_extent = first_frame.scene_extent
            self.active_sh_degree = first_frame.sh_degree
            self.current_point_count = first_frame.point_count
            self.sampled_point_count = first_frame.sampled_count
            self._insert_cpu_cache(0, first_frame)
            self._ensure_frame_on_gpu(0)
            self.prefetch_around(0)
    
    def _resolve_load_mode(self, requested_mode):
        if requested_mode != self.LOAD_MODE_AUTO:
            return requested_mode
        
        try:
            page_size = os.sysconf('SC_PAGE_SIZE')
            total_pages = os.sysconf('SC_PHYS_PAGES')
            total_ram_bytes = int(page_size * total_pages)
        except (ValueError, AttributeError, OSError):
            total_ram_bytes = None
        
        if total_ram_bytes is not None and self.total_sequence_bytes <= total_ram_bytes * 0.35:
            print("自动选择 preload_cpu 模式: 序列可放入系统内存，优先避免播放期读盘抖动")
            return self.LOAD_MODE_PRELOAD_CPU
        
        print("自动选择 stream 模式: 序列较大或无法判断内存容量，使用流式缓存")
        return self.LOAD_MODE_STREAM
    
    def _frame_path(self, frame_idx):
        return self.frame_paths[frame_idx % self.num_frames]

    def _frame_cache_path(self, frame_idx):
        if self.cache_frame_paths is None:
            return None
        return self.cache_frame_paths[frame_idx % self.num_frames]
    
    def _update_avg_stat(self, key, value, alpha=0.1):
        current = self.load_stats.get(key, 0.0)
        if current <= 0.0:
            self.load_stats[key] = float(value)
        else:
            self.load_stats[key] = alpha * float(value) + (1 - alpha) * current

    def _load_frame_cpu_sync(self, frame_idx, verbose=None):
        load_start = time.time()
        if self.cache_frame_paths is not None:
            frame = GaussianFrame.from_cache(
                self._frame_cache_path(frame_idx),
                pin_memory=self.pin_memory,
                verbose=self.verbose_frame_loads if verbose is None else verbose,
            )
        else:
            frame = GaussianFrame.from_ply(
                self._frame_path(frame_idx),
                sh_degree=self.sh_degree,
                pin_memory=self.pin_memory,
                max_gaussians=self.max_gaussians,
                verbose=self.verbose_frame_loads if verbose is None else verbose,
            )
        self._update_avg_stat('cpu_load_time_avg', time.time() - load_start)
        return frame
    
    def _insert_cpu_cache(self, frame_idx, frame):
        self.cpu_cache[frame_idx] = frame
        self.cpu_cache.move_to_end(frame_idx)
        while len(self.cpu_cache) > self.cpu_cache_size:
            self.cpu_cache.popitem(last=False)
    
    def _insert_gpu_cache(self, frame_idx, frame):
        self.gpu_cache[frame_idx] = frame
        self.gpu_cache.move_to_end(frame_idx)
        while len(self.gpu_cache) > self.gpu_cache_size:
            self.gpu_cache.popitem(last=False)

    def _resolve_pending_gpu(self, frame_idx, block=False):
        entry = self.pending_gpu.get(frame_idx)
        if entry is None:
            return None

        frame, _frame_cpu_ref, ready_event = entry
        if ready_event is not None:
            if block:
                ready_event.synchronize()
            elif not ready_event.query():
                return None

        self.pending_gpu.pop(frame_idx, None)
        self._insert_gpu_cache(frame_idx, frame)
        return frame
    
    def _preload_all_frames_to_gpu(self):
        print("正在预加载所有帧到GPU内存...")
        self.gpu_cache_size = max(self.gpu_cache_size, self.num_frames)
        for i in range(self.num_frames):
            frame_cpu = self._load_frame_cpu_sync(i, verbose=(i == 0 and self.verbose_frame_loads))
            frame_gpu = frame_cpu.to_device("cuda", non_blocking=False)
            self._insert_gpu_cache(i, frame_gpu)
            if i == self.num_frames - 1 or (i + 1) % max(1, min(16, self.num_frames)) == 0:
                print(f"  预加载进度: {i + 1}/{self.num_frames}")
            if i == 0:
                self.scene_center = frame_gpu.scene_center.copy()
                self.scene_extent = frame_gpu.scene_extent
                self.active_sh_degree = frame_gpu.sh_degree
                self.current_point_count = frame_gpu.point_count
                self.sampled_point_count = frame_gpu.sampled_count
        print("预加载完成！")
    
    def _preload_all_frames_to_cpu(self):
        print("正在预加载所有帧到CPU内存...")
        self.cpu_cache_size = max(self.cpu_cache_size, self.num_frames)
        for i in range(self.num_frames):
            frame_cpu = self._load_frame_cpu_sync(i, verbose=(i == 0 and self.verbose_frame_loads))
            self._insert_cpu_cache(i, frame_cpu)
            if i == self.num_frames - 1 or (i + 1) % max(1, min(16, self.num_frames)) == 0:
                print(f"  CPU预加载进度: {i + 1}/{self.num_frames}")
            if i == 0:
                self.scene_center = frame_cpu.scene_center.copy()
                self.scene_extent = frame_cpu.scene_extent
                self.active_sh_degree = frame_cpu.sh_degree
                self.current_point_count = frame_cpu.point_count
                self.sampled_point_count = frame_cpu.sampled_count
        print("CPU预加载完成，播放时只做GPU换帧")
    
    def shutdown(self):
        if self.executor is not None:
            try:
                self.executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                self.executor.shutdown(wait=False)
            self.executor = None
    
    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
    
    def set_fps(self, fps):
        self.fps = max(0.1, min(float(fps), 240.0))  # 限制在0.1-240 FPS范围
        self.last_update_time = time.time()
    
    def adjust_fps(self, delta):
        self.set_fps(self.fps + delta)
        return self.fps
    
    def set_fps_preset(self, preset_idx):
        """切换到预设FPS"""
        if 0 <= preset_idx < len(PLAYBACK_FPS_PRESETS):
            self.set_fps(PLAYBACK_FPS_PRESETS[preset_idx])
        return self.fps
    
    def cycle_fps_preset(self, forward=True):
        """循环切换播放FPS预设"""
        current_fps = self.fps
        presets = PLAYBACK_FPS_PRESETS
        
        # 找到最接近当前FPS的预设
        closest_idx = 0
        min_diff = abs(presets[0] - current_fps)
        for i, preset in enumerate(presets):
            diff = abs(preset - current_fps)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        
        # 切换到下一个/上一个预设
        if forward:
            next_idx = (closest_idx + 1) % len(presets)
        else:
            next_idx = (closest_idx - 1) % len(presets)
        
        self.set_fps(presets[next_idx])
        return self.fps
    
    def toggle_play(self):
        self.playing = not self.playing
        self.last_update_time = time.time()
        return self.playing
    
    def should_update(self):
        """检查是否应该更新到下一帧"""
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        if elapsed >= 1.0 / self.fps:
            self.last_update_time = current_time
            return True
        return False
    
    def consume_due_frames(self):
        """根据目标播放FPS计算当前应前进多少帧，用于掉帧时保持时间线"""
        current_time = time.time()
        frame_interval = 1.0 / self.fps
        elapsed = current_time - self.last_update_time
        if elapsed < frame_interval:
            return 0
        steps = max(1, int(elapsed / frame_interval))
        self.last_update_time += steps * frame_interval
        if current_time - self.last_update_time > frame_interval * 2:
            self.last_update_time = current_time
        return steps
    
    def _resolve_pending_cpu(self, frame_idx):
        future = self.pending_cpu.get(frame_idx)
        if future is None:
            return None
        frame = future.result()
        del self.pending_cpu[frame_idx]
        self._insert_cpu_cache(frame_idx, frame)
        return frame
    
    def _schedule_cpu_prefetch(self, frame_idx):
        if self.executor is None:
            return
        frame_idx %= self.num_frames
        if frame_idx in self.cpu_cache or frame_idx in self.gpu_cache or frame_idx in self.pending_cpu:
            return
        self.pending_cpu[frame_idx] = self.executor.submit(
            self._load_frame_cpu_sync,
            frame_idx,
            False,
        )

    def _effective_prefetch_count(self):
        if self.num_frames <= 1:
            return 0

        count = self.prefetch_count
        if self.adaptive_prefetch and self.fps > 0:
            est_load = max(
                self.load_stats.get('load_time_avg', 0.0),
                self.load_stats.get('cpu_load_time_avg', 0.0)
                + self.load_stats.get('gpu_upload_time_avg', 0.0),
            )
            if est_load > 0.0:
                count = max(count, int(math.ceil(est_load * self.fps)) + self.io_workers)

        if self.dynamic_cache:
            window_limit = self.cpu_cache_size + self.max_pending_cpu - 1
        else:
            window_limit = self.prefetch_count

        count = min(count, max(0, window_limit))
        return min(self.num_frames - 1, count)

    def _get_prefetch_targets(self, center_idx):
        forward_count = self._effective_prefetch_count()
        backward_count = min(self.backward_prefetch, self.num_frames - 1)
        if forward_count <= 0 and backward_count <= 0:
            return []

        targets = []
        if self.play_direction >= 0:
            primary_offsets = [center_idx + offset for offset in range(1, forward_count + 1)]
            secondary_offsets = [center_idx - offset for offset in range(1, backward_count + 1)]
        else:
            primary_offsets = [center_idx - offset for offset in range(1, forward_count + 1)]
            secondary_offsets = [center_idx + offset for offset in range(1, backward_count + 1)]

        for raw_idx in primary_offsets + secondary_offsets:
            frame_idx = raw_idx % self.num_frames
            if frame_idx not in targets and frame_idx != center_idx:
                targets.append(frame_idx)
        return targets

    def _drop_stale_prefetch(self, keep_targets):
        keep = set(keep_targets)
        keep.add(self.current_frame)
        for frame_idx, future in list(self.pending_cpu.items()):
            if frame_idx in keep:
                continue
            if future.cancel():
                self.pending_cpu.pop(frame_idx, None)

    def _fill_cpu_prefetch_queue(self):
        if self.load_mode != self.LOAD_MODE_STREAM:
            return []

        targets = self._get_prefetch_targets(self.current_frame)
        self._drop_stale_prefetch(targets)

        for frame_idx in targets:
            if len(self.pending_cpu) >= self.max_pending_cpu:
                break
            self._schedule_cpu_prefetch(frame_idx)
        return targets
    
    def service_prefetch(self):
        for frame_idx, future in list(self.pending_cpu.items()):
            if future.done():
                try:
                    frame = future.result()
                except Exception as exc:
                    print(f"预取失败(frame {frame_idx}): {exc}")
                else:
                    self._insert_cpu_cache(frame_idx, frame)
                finally:
                    self.pending_cpu.pop(frame_idx, None)
        
        if self.load_mode != self.LOAD_MODE_STREAM:
            return

        targets = self._fill_cpu_prefetch_queue()
        for frame_idx in targets:
            if frame_idx in self.cpu_cache and frame_idx not in self.gpu_cache and frame_idx not in self.pending_gpu:
                if len(self.gpu_cache) + len(self.pending_gpu) >= self.gpu_cache_size:
                    break
                self._start_gpu_prefetch(frame_idx)
    
    def _start_gpu_prefetch(self, frame_idx):
        if self.prefetch_stream is None:
            return
        frame_cpu = self.cpu_cache.get(frame_idx)
        if frame_cpu is None:
            return
        if frame_idx in self.gpu_cache or frame_idx in self.pending_gpu:
            return
        with torch.cuda.stream(self.prefetch_stream):
            frame_gpu = frame_cpu.to_device("cuda", non_blocking=frame_cpu.is_pinned)
            ready_event = torch.cuda.Event(blocking=False)
            ready_event.record()
        # 保留 CPU frame 引用直到异步 H2D 传输真正完成
        self.pending_gpu[frame_idx] = (frame_gpu, frame_cpu, ready_event)

    def request_frame(self, frame_idx, prefer_device=None):
        frame_idx %= self.num_frames
        self.service_prefetch()

        if prefer_device == "cpu":
            if frame_idx in self.cpu_cache:
                return
            future = self.pending_cpu.get(frame_idx)
            if future is not None:
                if future.done():
                    self._resolve_pending_cpu(frame_idx)
                return
            if self.executor is not None:
                self._schedule_cpu_prefetch(frame_idx)
            return

        if frame_idx in self.gpu_cache:
            return
        if self._resolve_pending_gpu(frame_idx, block=False) is not None:
            return

        if frame_idx in self.cpu_cache:
            self._start_gpu_prefetch(frame_idx)
            return

        future = self.pending_cpu.get(frame_idx)
        if future is not None:
            if future.done():
                self._resolve_pending_cpu(frame_idx)
                self._start_gpu_prefetch(frame_idx)
            return

        if self.executor is not None:
            self._schedule_cpu_prefetch(frame_idx)

    def is_frame_ready(self, frame_idx, prefer_device=None):
        frame_idx %= self.num_frames
        self.request_frame(frame_idx, prefer_device=prefer_device)

        if prefer_device == "cpu":
            return frame_idx in self.cpu_cache

        if frame_idx in self.gpu_cache:
            return True
        return self._resolve_pending_gpu(frame_idx, block=False) is not None
    
    def _resolve_cpu_frame(self, frame_idx):
        frame_idx %= self.num_frames
        if frame_idx in self.cpu_cache:
            frame = self.cpu_cache[frame_idx]
            self.cpu_cache.move_to_end(frame_idx)
            self.update_load_stats(hit=True)
            return frame
        if frame_idx in self.pending_cpu:
            self.update_load_stats(hit=True)
            return self._resolve_pending_cpu(frame_idx)
        load_start = time.time()
        frame = self._load_frame_cpu_sync(frame_idx)
        self._insert_cpu_cache(frame_idx, frame)
        self.update_load_stats(hit=False, load_time=time.time() - load_start)
        return frame
    
    def _ensure_frame_on_gpu(self, frame_idx):
        frame_idx %= self.num_frames
        self.service_prefetch()
        
        if frame_idx in self.gpu_cache:
            frame = self.gpu_cache[frame_idx]
            self.gpu_cache.move_to_end(frame_idx)
            self.update_load_stats(hit=True)
            return frame
        
        load_start = time.time()
        
        if frame_idx in self.pending_gpu:
            frame = self._resolve_pending_gpu(frame_idx, block=True)
            self.update_load_stats(hit=True)  # 预取命中算缓存命中
            return frame
        
        frame_cpu = self._resolve_cpu_frame(frame_idx)
        upload_start = time.time()
        frame_gpu = frame_cpu.to_device("cuda", non_blocking=frame_cpu.is_pinned)
        self._insert_gpu_cache(frame_idx, frame_gpu)
        
        self._update_avg_stat('gpu_upload_time_avg', time.time() - upload_start)
        load_time = time.time() - load_start
        self.update_load_stats(hit=False, load_time=load_time)
        return frame_gpu
    
    def prefetch_around(self, frame_idx):
        if self.load_mode != self.LOAD_MODE_STREAM:
            return
        self.current_frame = frame_idx % self.num_frames
        self.service_prefetch()
    
    def get_current_frame_data(self, prefer_device=None):
        """获取当前帧的数据"""
        if prefer_device == "cpu":
            frame = self._resolve_cpu_frame(self.current_frame)
        else:
            frame = self._ensure_frame_on_gpu(self.current_frame)
        self.active_sh_degree = frame.sh_degree
        self.current_point_count = frame.point_count
        self.sampled_point_count = frame.sampled_count
        self.prefetch_around(self.current_frame)
        return frame
    
    def next_frame(self):
        self.play_direction = 1
        self.current_frame = (self.current_frame + 1) % self.num_frames
        self.prefetch_around(self.current_frame)
    
    def prev_frame(self):
        self.play_direction = -1
        self.current_frame = (self.current_frame - 1) % self.num_frames
        self.prefetch_around(self.current_frame)
    
    def set_frame(self, frame_idx):
        raw_idx = int(frame_idx)
        next_idx = raw_idx % self.num_frames
        if raw_idx != self.current_frame:
            self.play_direction = 1 if raw_idx > self.current_frame else -1
        self.current_frame = next_idx
        self.prefetch_around(self.current_frame)
    
    def get_cache_status(self):
        return (
            f"{self.load_mode}/{self.load_backend} | GPU {len(self.gpu_cache)}/{self.gpu_cache_size} | "
            f"CPU {len(self.cpu_cache)}/{self.cpu_cache_size} (+{len(self.pending_cpu)}) | "
            f"IO x{self.io_workers}"
        )
    
    def get_load_stats(self):
        """获取加载统计信息"""
        total = self.load_stats['cache_hits'] + self.load_stats['cache_misses']
        hit_rate = (self.load_stats['cache_hits'] / total * 100) if total > 0 else 0.0
        return {
            'hit_rate': hit_rate,
            'avg_load_time': self.load_stats['load_time_avg'],
            'avg_cpu_load_time': self.load_stats['cpu_load_time_avg'],
            'avg_gpu_upload_time': self.load_stats['gpu_upload_time_avg'],
            'total_accesses': total,
        }
    
    def update_load_stats(self, hit=True, load_time=0.0):
        """更新加载统计"""
        if hit:
            self.load_stats['cache_hits'] += 1
        else:
            self.load_stats['cache_misses'] += 1
            self._update_avg_stat('load_time_avg', load_time)


class GaussianPointCloud:
    """加载和管理3D Gaussian点云数据"""
    
    def __init__(self, ply_path, sh_degree=3, device="cuda", max_gaussians=None):
        self.device = device
        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.ply_path = ply_path
        self.max_gaussians = max_gaussians
        self._reset_storage()
        self.load_ply(ply_path)
    
    @classmethod
    def from_frame(cls, frame, device="cuda"):
        pc = cls.__new__(cls)
        pc.device = device
        pc.ply_path = frame.source_path
        pc.max_gaussians = None
        pc._reset_storage()
        pc.apply_frame(frame)
        pc.max_sh_degree = frame.sh_degree
        pc.active_sh_degree = frame.sh_degree
        return pc

    def _reset_storage(self):
        self._xyz_buffer = None
        self._features_buffer = None
        self._scaling_buffer = None
        self._rotation_buffer = None
        self._opacity_buffer = None
        self._frame_ref = None
        self._xyz = None
        self._features = None
        self._scaling = None
        self._rotation = None
        self._opacity = None
        self._point_count = 0
        self._buffer_capacity = 0
        self._sample_index_cache = {}

    def _buffer_device(self):
        return torch.device(self.device)

    def _copy_into(self, dst, src):
        non_blocking = False
        if src.device.type == 'cuda':
            non_blocking = True
        elif src.device.type == 'cpu' and src.is_pinned():
            non_blocking = True
        dst.copy_(src, non_blocking=non_blocking)

    def _allocate_buffers(self, point_count, xyz_dim, features_shape, scaling_dim, rotation_dim, opacity_dim):
        extra_slack = max(16384, point_count // 20)
        capacity = max(point_count, point_count + extra_slack)
        device = self._buffer_device()

        self._xyz_buffer = torch.empty((capacity, xyz_dim), dtype=torch.float32, device=device)
        self._features_buffer = torch.empty((capacity, *features_shape), dtype=torch.float32, device=device)
        self._scaling_buffer = torch.empty((capacity, scaling_dim), dtype=torch.float32, device=device)
        self._rotation_buffer = torch.empty((capacity, rotation_dim), dtype=torch.float32, device=device)
        self._opacity_buffer = torch.empty((capacity, opacity_dim), dtype=torch.float32, device=device)
        self._buffer_capacity = capacity

    def _ensure_buffers(self, point_count, features_shape, scaling_shape, rotation_shape, opacity_shape):
        if self._xyz_buffer is None:
            self._allocate_buffers(
                point_count,
                3,
                features_shape,
                scaling_shape[0],
                rotation_shape[0],
                opacity_shape[0],
            )
            return

        if (
            point_count > self._buffer_capacity
            or self._features_buffer.shape[1:] != features_shape
            or self._scaling_buffer.shape[1:] != scaling_shape
            or self._rotation_buffer.shape[1:] != rotation_shape
            or self._opacity_buffer.shape[1:] != opacity_shape
            or self._xyz_buffer.device != self._buffer_device()
        ):
            self._allocate_buffers(
                point_count,
                3,
                features_shape,
                scaling_shape[0],
                rotation_shape[0],
                opacity_shape[0],
            )

    def _get_sample_indices(self, total_points, max_points, device):
        key = (int(total_points), int(max_points), str(device))
        indices = self._sample_index_cache.get(key)
        if indices is not None:
            return indices

        index_np = np.linspace(0, total_points - 1, num=max_points, dtype=np.int64)
        indices = torch.from_numpy(index_np)
        if device.type != 'cpu':
            indices = indices.to(device=device, non_blocking=True)
        self._sample_index_cache[key] = indices
        return indices
    
    def apply_frame(self, frame, max_points=None, sh_degree=None):
        src_xyz = frame.xyz
        src_features = frame.features
        src_scaling = frame.scaling
        src_rotation = frame.rotation
        src_opacity = frame.opacity

        point_count = frame.tensor_count
        if max_points is not None:
            max_points = max(1, int(max_points))
            if point_count > max_points:
                sample_indices = self._get_sample_indices(point_count, max_points, src_xyz.device)
                src_xyz = src_xyz.index_select(0, sample_indices)
                src_features = src_features.index_select(0, sample_indices)
                src_scaling = src_scaling.index_select(0, sample_indices)
                src_rotation = src_rotation.index_select(0, sample_indices)
                src_opacity = src_opacity.index_select(0, sample_indices)
                point_count = max_points

        active_sh_degree = frame.sh_degree
        if sh_degree is not None:
            active_sh_degree = max(0, min(int(sh_degree), frame.sh_degree))
            coeff_count = (active_sh_degree + 1) ** 2
            src_features = src_features[:, :coeff_count, :]

        if (
            max_points is None
            and sh_degree is None
            and src_xyz.device == self._buffer_device()
        ):
            self._frame_ref = frame
            self._point_count = point_count
            self._xyz = src_xyz
            self._features = src_features
            self._scaling = src_scaling
            self._rotation = src_rotation
            self._opacity = src_opacity
            self.scene_center = frame.scene_center.copy()
            self.scene_extent = frame.scene_extent
            self.active_sh_degree = active_sh_degree
            self.max_sh_degree = frame.sh_degree
            return

        self._frame_ref = None
        self._ensure_buffers(
            point_count,
            src_features.shape[1:],
            src_scaling.shape[1:],
            src_rotation.shape[1:],
            src_opacity.shape[1:],
        )

        self._copy_into(self._xyz_buffer[:point_count], src_xyz)
        self._copy_into(self._features_buffer[:point_count], src_features)
        self._copy_into(self._scaling_buffer[:point_count], src_scaling)
        self._copy_into(self._rotation_buffer[:point_count], src_rotation)
        self._copy_into(self._opacity_buffer[:point_count], src_opacity)

        self._point_count = point_count
        self._xyz = self._xyz_buffer[:point_count]
        self._features = self._features_buffer[:point_count]
        self._scaling = self._scaling_buffer[:point_count]
        self._rotation = self._rotation_buffer[:point_count]
        self._opacity = self._opacity_buffer[:point_count]
        self.scene_center = frame.scene_center.copy()
        self.scene_extent = frame.scene_extent
        self.active_sh_degree = active_sh_degree
        self.max_sh_degree = frame.sh_degree
    
    def reload(self, ply_path):
        """重新加载点云（用于序列播放）"""
        self.ply_path = ply_path
        self.load_ply(ply_path)
    
    def load_ply(self, path):
        """从PLY文件加载高斯点云"""
        frame = GaussianFrame.from_ply(
            path,
            sh_degree=self.max_sh_degree,
            pin_memory=False,
            max_gaussians=self.max_gaussians,
        )
        self.ply_path = path
        self.apply_frame(frame)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_scaling(self):
        return self._scaling
    
    @property
    def get_rotation(self):
        return self._rotation
    
    @property
    def get_opacity(self):
        return self._opacity
    
    @property
    def get_features(self):
        return self._features


class Camera:
    """相机类，与原始3DGS Camera保持一致"""
    
    def __init__(self, R, T, FoVx, FoVy, width, height, znear=0.01, zfar=100.0):
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = width
        self.image_height = height
        self.znear = znear
        self.zfar = zfar
        
        self._update_matrices()
    
    def _update_matrices(self):
        """更新变换矩阵"""
        self.world_view_transform = torch.tensor(
            getWorld2View2(self.R, self.T)
        ).transpose(0, 1).cuda()
        
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, 
            fovX=self.FoVx, fovY=self.FoVy
        ).transpose(0, 1).cuda()
        
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class InteractiveCamera:
    """交互式相机控制器 (SIBR风格)"""
    
    # 相机模式
    MODE_FPS = 0
    MODE_TRACKBALL = 1
    MODE_ORBIT = 2
    
    def __init__(self, width, height, cameras_info=None, scene_center=None, scene_extent=1.0):
        self.width = width
        self.height = height
        self.cameras_info = cameras_info if cameras_info else []
        self.current_camera_idx = -1
        
        self.scene_center = scene_center if scene_center is not None else np.zeros(3)
        self.scene_extent = scene_extent
        
        # 初始化相机参数
        self.znear = 0.01
        self.zfar = 100.0
        
        # 当前相机状态 - 相机在场景Z负方向，看向场景中心
        # 注意：初始位置在Z负方向，这样默认朝向（Z正）就能看到场景
        self.position = self.scene_center - np.array([0, 0, scene_extent * 1.5], dtype=np.float32)
        self.R = np.eye(3, dtype=np.float32)
        self.FoVx = math.radians(60)
        self.FoVy = compute_fovy_from_fovx(self.FoVx, width, height)
        
        print(f"  场景中心: {self.scene_center}")
        print(f"  场景范围: {scene_extent:.2f}")
        print(f"  相机初始位置: {self.position}")
        
        # SIBR风格速度控制
        self.move_speed = scene_extent * 0.002  # 移动速度 (WASD/QE)
        self.rot_speed = 0.02  # 旋转速度 (IJKL/UO)
        self.mouse_sensitivity = 0.003  # 鼠标灵敏度
        
        # 相机模式
        self.mode = self.MODE_FPS
        
        # Trackball参数
        self.trackball_center = self.scene_center.copy()
        self.trackball_radius = scene_extent * 0.5
    
    def _axis_angle_to_rotation(self, axis, angle):
        """将轴角表示转换为旋转矩阵 (Rodrigues公式)"""
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], dtype=np.float32)
        R = np.eye(3, dtype=np.float32) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
        return R
    
    def get_T(self):
        """计算T向量"""
        return (-self.R.T @ self.position).astype(np.float32)
    
    def get_camera(self):
        """获取当前相机对象"""
        return Camera(
            R=self.R.copy(),
            T=self.get_T(),
            FoVx=self.FoVx,
            FoVy=self.FoVy,
            width=self.width,
            height=self.height,
            znear=self.znear,
            zfar=self.zfar
        )
    
    def set_camera(self, idx):
        """切换到预设相机视角"""
        if 0 <= idx < len(self.cameras_info):
            cam = self.cameras_info[idx]
            self.current_camera_idx = idx
            
            self.R = cam['rotation'].copy()
            self.position = cam['position'].copy()
            
            # 根据相机分辨率调整FOV
            self.FoVx = focal2fov(cam['fx'], cam['width'])
            self.FoVy = compute_fovy_from_fovx(self.FoVx, self.width, self.height)
            
            print(f"切换到相机 {idx + 1}: {cam['name']}")
            return True
        return False
    
    def snap_to_nearest_camera(self):
        """跳转到最近的相机视角 (SIBR P键功能)"""
        if len(self.cameras_info) == 0:
            return
        
        min_dist = float('inf')
        nearest_idx = 0
        for i, cam in enumerate(self.cameras_info):
            dist = np.linalg.norm(self.position - cam['position'])
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        self.set_camera(nearest_idx)
    
    def reset(self):
        """重置相机到初始位置"""
        self.position = self.scene_center - np.array([0, 0, self.scene_extent * 1.5])
        self.R = np.eye(3, dtype=np.float32)
        self.FoVx = math.radians(60)
        self.FoVy = compute_fovy_from_fovx(self.FoVx, self.width, self.height)
        self.current_camera_idx = -1
        print("视角已重置")
    
    def switch_mode(self, mode):
        """切换相机模式"""
        if self.mode == mode:
            mode = self.MODE_FPS  # 再次按下则回到FPS模式
        
        self.mode = mode
        mode_names = {
            self.MODE_FPS: "FPS模式",
            self.MODE_TRACKBALL: "Trackball模式",
            self.MODE_ORBIT: "Orbit模式"
        }
        print(f"切换到 {mode_names.get(mode, 'FPS模式')}")
    
    def next_camera(self):
        if len(self.cameras_info) > 0:
            next_idx = (self.current_camera_idx + 1) % len(self.cameras_info)
            self.set_camera(next_idx)
    
    def prev_camera(self):
        if len(self.cameras_info) > 0:
            prev_idx = (self.current_camera_idx - 1) % len(self.cameras_info)
            self.set_camera(prev_idx)
    
    def resize(self, width, height):
        """调整渲染分辨率"""
        if width > 0 and height > 0:
            self.width = width
            self.height = height
            # 保持水平FOV不变，只按宽高比更新垂直FOV
            self.FoVy = compute_fovy_from_fovx(self.FoVx, width, height)
    
    # ============================================
    # SIBR FPS模式移动
    # 注意：3DGS中world_view_transform使用R.T
    # 所以相机轴在世界坐标系中的方向是R的列：
    #   R[:, 0] = 相机X轴 (right)
    #   R[:, 1] = 相机Y轴 (down)  
    #   R[:, 2] = 相机Z轴 (forward/look direction)
    # ============================================
    
    def move_forward(self, delta):
        """W/S - 沿视线方向移动
        SIBR: move.z() -= W (前进), move.z() += S (后退)
        """
        forward = self.R[:, 2]  # 相机Z轴（列）
        self.position += forward * delta * self.move_speed  # 修正方向
        self.current_camera_idx = -1
    
    def move_right(self, delta):
        """A/D - 左右移动
        SIBR: move.x() -= A (左移), move.x() += D (右移)
        """
        right = self.R[:, 0]  # 相机X轴（列）
        self.position += right * delta * self.move_speed
        self.current_camera_idx = -1
    
    def move_up(self, delta):
        """Q/E - 上下移动
        SIBR: move.y() -= Q (下降), move.y() += E (上升)
        在COLMAP中Y轴向下，所以down = R[:,1], up = -R[:,1]
        """
        down = self.R[:, 1]  # 相机Y轴（列）
        self.position -= down * delta * self.move_speed  # y-= 下降方向取反 = 上升
        self.current_camera_idx = -1
    
    # ============================================
    # SIBR FPS模式旋转 (在相机自身坐标系中旋转)
    # ============================================
    
    def rotate_pitch(self, delta):
        """I/K - 上下看 (绕相机X轴旋转)
        SIBR: pivot[0] += I (向上看), pivot[0] -= K (向下看)
        """
        angle = delta * self.rot_speed  # 修正方向
        axis_cam = np.array([1, 0, 0], dtype=np.float32)
        dR_cam = self._axis_angle_to_rotation(axis_cam, angle)
        self.R = self.R @ dR_cam  # 右乘在相机坐标系中旋转
        self.current_camera_idx = -1
    
    def rotate_yaw(self, delta):
        """J/L - 左右看 (绕相机Y轴旋转)
        SIBR: pivot[1] += J (向左看), pivot[1] -= L (向右看)
        """
        angle = delta * self.rot_speed
        axis_cam = np.array([0, 1, 0], dtype=np.float32)
        dR_cam = self._axis_angle_to_rotation(axis_cam, angle)
        self.R = self.R @ dR_cam
        self.current_camera_idx = -1
    
    def rotate_roll(self, delta):
        """U/O - 滚转 (绕相机Z轴旋转)
        SIBR: pivot[2] += U, pivot[2] -= O
        """
        angle = delta * self.rot_speed
        axis_cam = np.array([0, 0, 1], dtype=np.float32)
        dR_cam = self._axis_angle_to_rotation(axis_cam, angle)
        self.R = self.R @ dR_cam
        self.current_camera_idx = -1
    
    # === Trackball模式 ===
    def trackball_rotate(self, dx, dy):
        """Trackball球面旋转 (鼠标左键中心区域)"""
        angle_yaw = dx * self.mouse_sensitivity
        angle_pitch = -dy * self.mouse_sensitivity
        
        # Yaw: 绕相机Y轴
        axis_yaw = np.array([0, 1, 0], dtype=np.float32)
        dR_yaw = self._axis_angle_to_rotation(axis_yaw, angle_yaw)
        
        # Pitch: 绕相机X轴
        axis_pitch = np.array([1, 0, 0], dtype=np.float32)
        dR_pitch = self._axis_angle_to_rotation(axis_pitch, angle_pitch)
        
        # 应用旋转
        self.R = self.R @ dR_yaw @ dR_pitch
        
        # 更新位置使其绕中心旋转
        offset = self.position - self.trackball_center
        dist = np.linalg.norm(offset)
        
        forward = self.R[:, 2]  # 相机前方(列)
        self.position = self.trackball_center - forward * dist
        self.current_camera_idx = -1
    
    def trackball_roll(self, dx):
        """Trackball滚转 (鼠标左键边缘区域)"""
        angle = dx * self.mouse_sensitivity
        axis_cam = np.array([0, 0, 1], dtype=np.float32)
        dR = self._axis_angle_to_rotation(axis_cam, angle)
        self.R = self.R @ dR
        self.current_camera_idx = -1
    
    def trackball_pan(self, dx, dy):
        """Trackball平移 (鼠标右键中心区域)"""
        right = self.R[:, 0]  # 相机X轴(列)
        up = -self.R[:, 1]    # 相机Y向下，up取反
        move = right * dx * self.move_speed * 0.1 + up * dy * self.move_speed * 0.1
        self.position += move
        self.trackball_center += move
        self.current_camera_idx = -1
    
    def trackball_zoom(self, delta):
        """Trackball缩放 (鼠标右键边缘区域 或 滚轮)"""
        forward = self.R[:, 2]  # 相机Z轴(列)
        self.position -= forward * delta * self.move_speed
        self.current_camera_idx = -1
    
    def zoom(self, delta):
        """缩放"""
        if self.mode == self.MODE_TRACKBALL:
            self.trackball_zoom(delta * 5)
        else:
            self.move_forward(delta * 5)


class GaussianRenderer:
    """3D Gaussian Splatting渲染器"""
    
    def __init__(self, pc, bg_color=[0, 0, 0]):
        self.pc = pc
        self.bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self._means2d_buffer = None
        self._means2d_count = -1
    
    def _get_screenspace_buffer(self):
        point_count = self.pc.get_xyz.shape[0]
        if (
            self._means2d_buffer is None
            or self._means2d_count != point_count
            or self._means2d_buffer.device != self.pc.get_xyz.device
        ):
            self._means2d_buffer = torch.zeros(
                (point_count, 3),
                dtype=torch.float32,
                device=self.pc.get_xyz.device,
            )
            self._means2d_count = point_count
        else:
            self._means2d_buffer.zero_()
        return self._means2d_buffer
        
    def render(self, camera, resolution_scale=1.0):
        """渲染一帧"""
        with torch.inference_mode():
            cam = camera.get_camera()
            scale = float(max(0.1, min(resolution_scale, 1.0)))
            image_width = max(1, int(round(cam.image_width * scale)))
            image_height = max(1, int(round(cam.image_height * scale)))
            
            screenspace_points = self._get_screenspace_buffer()
            
            raster_settings = GaussianRasterizationSettings(
                image_height=image_height,
                image_width=image_width,
                tanfovx=math.tan(cam.FoVx * 0.5),
                tanfovy=math.tan(cam.FoVy * 0.5),
                bg=self.bg_color,
                scale_modifier=1.0,
                viewmatrix=cam.world_view_transform,
                projmatrix=cam.full_proj_transform,
                sh_degree=self.pc.active_sh_degree,
                campos=cam.camera_center,
                prefiltered=False,
                debug=False,
                antialiasing=False
            )
            
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            
            # 兼容不同版本的rasterizer（有的返回2个值，有的返回3个）
            result = rasterizer(
                means3D=self.pc.get_xyz,
                means2D=screenspace_points,
                shs=self.pc.get_features,
                colors_precomp=None,
                opacities=self.pc.get_opacity,
                scales=self.pc.get_scaling,
                rotations=self.pc.get_rotation,
                cov3D_precomp=None
            )
            
            # 处理不同版本的返回值
            if len(result) == 2:
                rendered_image, radii = result
            else:
                rendered_image, radii, depth = result
            
            return rendered_image.clamp(0, 1)


class PygameViewer:
    """基于Pygame的交互式查看器 (SIBR风格控制)
    
    采用类似 SIBR 的设计：
    - 渲染分辨率固定，可独立于窗口分辨率调整
    - 窗口可以自由调整大小
    - 渲染图像缩放适应窗口
    """
    
    def __init__(
        self,
        renderer,
        camera,
        sequence_manager=None,
        title="3DGS Python Viewer",
        window_size=None,
        auto_fit_window=False,
    ):
        pygame.init()
        
        self.renderer = renderer
        self.camera = camera
        self.sequence_manager = sequence_manager
        self.point_cloud = renderer.pc
        self.title = title
        
        self.render_width = camera.width
        self.render_height = camera.height
        self.render_resolution_label = self._get_resolution_label(self.render_width, self.render_height)
        self.current_resolution_idx = self._find_resolution_preset_idx(self.render_width, self.render_height)
        
        if window_size is None:
            window_size = (self.render_width, self.render_height)
        if auto_fit_window:
            window_size = self._fit_window_size(*window_size)
        self.window_width, self.window_height = window_size
        
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
        pygame.display.set_caption(title)
        
        self.clock = pygame.time.Clock()
        
        self.running = True
        self.left_mouse_pressed = False
        self.right_mouse_pressed = False
        self.middle_mouse_pressed = False
        self.last_mouse_pos = None
        self.screenshot_count = 0
        
        # Trackball区域比例 (中心75%为球面旋转区域)
        self.trackball_ratio = 0.75
        
        # 显示帮助信息开关
        self.show_help = False
    
    def _get_resolution_label(self, width, height):
        """根据分辨率返回对应的标签"""
        for label, (w, h) in RUNTIME_RESOLUTION_PRESETS:
            if w == width and h == height:
                return label.upper()
        return f"{width}x{height}"
    
    def _find_resolution_preset_idx(self, width, height):
        """找到当前分辨率对应的预设索引，未找到返回-1"""
        for i, (label, (w, h)) in enumerate(RUNTIME_RESOLUTION_PRESETS):
            if w == width and h == height:
                return i
        return -1
    
    def cycle_resolution_preset(self, forward=True):
        """循环切换渲染分辨率预设"""
        presets = RUNTIME_RESOLUTION_PRESETS
        if forward:
            self.current_resolution_idx = (self.current_resolution_idx + 1) % len(presets)
        else:
            self.current_resolution_idx = (self.current_resolution_idx - 1) % len(presets)
        
        label, (width, height) = presets[self.current_resolution_idx]
        self.set_render_resolution(width, height, label=label.upper())
    
    def _fit_window_size(self, width, height):
        display_info = pygame.display.Info()
        return fit_size_within_bounds(width, height, display_info.current_w, display_info.current_h)
    
    def set_window_size(self, width, height):
        self.window_width = max(320, int(width))
        self.window_height = max(240, int(height))
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
    
    def set_render_resolution(self, width, height, label=None):
        self.render_width = max(64, int(width))
        self.render_height = max(64, int(height))
        self.render_resolution_label = label or f"{self.render_width}x{self.render_height}"
        self.camera.resize(self.render_width, self.render_height)
        print(f"渲染分辨率切换为: {self.render_resolution_label} ({self.render_width}x{self.render_height})")
    
    def adjust_playback_fps(self, delta):
        if not self.sequence_manager:
            return
        new_fps = self.sequence_manager.adjust_fps(delta)
        print(f"播放FPS调整为: {new_fps:.1f}")
    
    def cycle_playback_fps(self, forward=True):
        """循环切换播放FPS预设"""
        if not self.sequence_manager:
            return
        new_fps = self.sequence_manager.cycle_fps_preset(forward)
        print(f"播放FPS切换为: {new_fps:.0f}")
    
    def apply_runtime_resolution_shortcut(self, key):
        preset = RUNTIME_RESOLUTION_KEYS.get(key)
        if preset is None:
            return False
        label, (width, height) = preset
        self.set_render_resolution(width, height, label=label.upper())
        self.current_resolution_idx = self._find_resolution_preset_idx(width, height)
        return True
    
    def is_in_trackball_center(self, pos):
        """检查鼠标是否在Trackball中心区域"""
        cx, cy = self.window_width / 2, self.window_height / 2
        radius = min(self.window_width, self.window_height) / 2 * self.trackball_ratio
        dx = pos[0] - cx
        dy = pos[1] - cy
        return (dx * dx + dy * dy) < (radius * radius)
        
    def process_events(self):
        """处理输入事件 (SIBR风格)"""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == VIDEORESIZE:
                self.set_window_size(event.w, event.h)
            elif hasattr(pygame, 'WINDOWRESIZED') and event.type == pygame.WINDOWRESIZED:
                self.window_width = max(320, event.x)
                self.window_height = max(240, event.y)
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
                elif self.apply_runtime_resolution_shortcut(event.key):
                    pass
                elif event.key == K_r:
                    # R键循环切换分辨率预设
                    self.cycle_resolution_preset(forward=True)
                elif event.key == K_f and self.sequence_manager:
                    # F键循环切换播放FPS预设
                    self.cycle_playback_fps(forward=True)
                elif event.key in (K_MINUS, PYGAME_K_KP_MINUS) and self.sequence_manager:
                    self.adjust_playback_fps(-5.0)
                elif event.key in (K_EQUALS, PYGAME_K_PLUS, PYGAME_K_KP_PLUS) and self.sequence_manager:
                    self.adjust_playback_fps(5.0)
                elif event.key == K_h:
                    # H键切换帮助显示
                    self.show_help = not self.show_help
                elif event.key == K_m:
                    self.save_screenshot()
                elif event.key == K_y:
                    self.camera.switch_mode(InteractiveCamera.MODE_TRACKBALL)
                elif event.key == K_b:
                    self.camera.switch_mode(InteractiveCamera.MODE_ORBIT)
                elif event.key == K_p:
                    self.camera.snap_to_nearest_camera()
                elif event.key == K_n:
                    self.camera.next_camera()
                elif event.key == K_SPACE and self.sequence_manager:
                    self.sequence_manager.toggle_play()
                elif event.key == K_LEFT and self.sequence_manager:
                    self.sequence_manager.prev_frame()
                    self.reload_current_frame()
                elif event.key == K_RIGHT and self.sequence_manager:
                    self.sequence_manager.next_frame()
                    self.reload_current_frame()
                elif event.key == K_HOME and self.sequence_manager:
                    self.sequence_manager.set_frame(0)
                    self.reload_current_frame()
                elif event.key == K_END and self.sequence_manager:
                    self.sequence_manager.set_frame(self.sequence_manager.num_frames - 1)
                    self.reload_current_frame()
                    
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键
                    self.left_mouse_pressed = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 3:  # 右键
                    self.right_mouse_pressed = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 2:  # 中键
                    self.middle_mouse_pressed = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4:  # 滚轮上
                    self.camera.zoom(1)
                elif event.button == 5:  # 滚轮下
                    self.camera.zoom(-1)
                    
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.left_mouse_pressed = False
                elif event.button == 3:
                    self.right_mouse_pressed = False
                elif event.button == 2:
                    self.middle_mouse_pressed = False
                if not (self.left_mouse_pressed or self.right_mouse_pressed or self.middle_mouse_pressed):
                    self.last_mouse_pos = None
                    
            elif event.type == MOUSEMOTION:
                if self.last_mouse_pos:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    
                    if self.camera.mode == InteractiveCamera.MODE_TRACKBALL:
                        # Trackball模式鼠标控制
                        in_center = self.is_in_trackball_center(self.last_mouse_pos)
                        if self.left_mouse_pressed:
                            if in_center:
                                self.camera.trackball_rotate(dx, dy)
                            else:
                                self.camera.trackball_roll(dx)
                        elif self.right_mouse_pressed:
                            if in_center:
                                self.camera.trackball_pan(-dx, dy)
                            else:
                                self.camera.trackball_zoom(-dy * 0.5)
                    else:
                        # FPS模式: 鼠标左键=X移动, 右键=Y移动, 中键=Z移动 (SIBR风格)
                        if self.left_mouse_pressed:
                            self.camera.move_right(dx * 0.1)
                        if self.right_mouse_pressed:
                            self.camera.move_up(-dy * 0.1)
                        if self.middle_mouse_pressed:
                            self.camera.move_forward(-dy * 0.1)
                    
                    self.last_mouse_pos = event.pos
        
        # 键盘持续按下检测 (SIBR FPS模式)
        keys = pygame.key.get_pressed()
        
        if keys[K_LCTRL] or keys[K_RCTRL]:
            return
        
        if keys[K_w]:
            self.camera.move_forward(1)
        if keys[K_s]:
            self.camera.move_forward(-1)
        if keys[K_a]:
            self.camera.move_right(-1)
        if keys[K_d]:
            self.camera.move_right(1)
        if keys[K_q]:
            self.camera.move_up(-1)
        if keys[K_e]:
            self.camera.move_up(1)
        
        if keys[K_i]:
            self.camera.rotate_pitch(1)
        if keys[K_k]:
            self.camera.rotate_pitch(-1)
        if keys[K_j]:
            self.camera.rotate_yaw(-1)
        if keys[K_l]:
            self.camera.rotate_yaw(1)
        if keys[K_u]:
            self.camera.rotate_roll(1)
        if keys[K_o]:
            self.camera.rotate_roll(-1)
            
    def reload_current_frame(self):
        """切换到当前帧（优先命中GPU缓存，否则从CPU缓存/磁盘流式加载）"""
        if self.sequence_manager:
            frame_data = self.sequence_manager.get_current_frame_data()
            self.point_cloud.apply_frame(frame_data)
    
    def save_screenshot(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}_{self.screenshot_count}.png"
        pygame.image.save(self.screen, filename)
        print(f"截图已保存: {filename}")
        self.screenshot_count += 1
        
    def run(self):
        num_cams = len(self.camera.cameras_info)
        print("\n" + "="*60)
        print("  3DGS Python交互式查看器 (支持4K/2K/1080p渲染分辨率)")
        print("="*60)
        print("\n=== 渲染分辨率控制 ===")
        print("  1/2/3/4 - 直接切换 720p/1080p/2K/4K")
        print("  R - 循环切换分辨率预设")
        print("  (窗口大小可自由拖动，不影响渲染分辨率)")
        print("\n=== FPS模式 (默认) ===")
        print("  W/A/S/D - 前进/左移/后退/右移")
        print("  Q/E - 下降/上升")
        print("  I/K - 向上/向下看")
        print("  J/L - 向左/向右看")
        print("  U/O - 左滚/右滚")
        print("  鼠标左键 - X移动")
        print("  鼠标右键 - Y移动")
        print("  鼠标中键 - Z移动")
        print("\n=== Trackball模式 (按Y切换) ===")
        print("  鼠标左键(中心) - 球面旋转")
        print("  鼠标左键(边缘) - 滚转")
        print("  鼠标右键(中心) - 平面平移")
        print("  鼠标右键(边缘) - 前后移动")
        print("  滚轮 - 缩放")
        print("\n=== 相机控制 ===")
        print("  P - 跳转到最近相机")
        print("  N - 下一个相机")
        if num_cams > 0:
            print(f"  (共{num_cams}个预设相机)")
        print("\n=== 其他 ===")
        print("  Y - 切换FPS/Trackball模式")
        print("  B - 切换FPS/Orbit模式")
        print("  H - 切换帮助信息显示")
        print("  M - 截图保存")
        print("  ESC - 退出")
        if self.sequence_manager:
            print("\n=== 序列播放 ===")
            print("  空格 - 播放/暂停")
            print("  左/右箭头 - 上一帧/下一帧")
            print("  HOME/END - 第一帧/最后一帧")
            print("  F - 循环切换播放FPS预设 (10/15/24/30/60/120)")
            print("  -/+ - 微调播放FPS (±5)")
            print(f"  (共{self.sequence_manager.num_frames}帧)")
        print("="*60 + "\n")
        
        font = pygame.font.Font(None, 30)
        small_font = pygame.font.Font(None, 24)
        
        while self.running:
            self.process_events()
            if self.sequence_manager:
                self.sequence_manager.service_prefetch()
            
            if self.sequence_manager and self.sequence_manager.playing:
                due_frames = self.sequence_manager.consume_due_frames()
                if due_frames > 0:
                    self.sequence_manager.set_frame(self.sequence_manager.current_frame + due_frames)
                    self.reload_current_frame()
            
            current_size = self.screen.get_size()
            self.window_width = current_size[0]
            self.window_height = current_size[1]
            
            render_aspect = self.render_width / self.render_height
            window_aspect = self.window_width / self.window_height
            if window_aspect > render_aspect:
                display_height = self.window_height
                display_width = int(display_height * render_aspect)
                offset_x = (self.window_width - display_width) // 2
                offset_y = 0
            else:
                display_width = self.window_width
                display_height = int(display_width / render_aspect)
                offset_x = 0
                offset_y = (self.window_height - display_height) // 2
            
            rendered = self.renderer.render(self.camera)
            
            image = rendered.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
            image = np.ascontiguousarray(image)
            
            render_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
            if display_width != self.render_width or display_height != self.render_height:
                render_surface = pygame.transform.scale(render_surface, (display_width, display_height))
            
            self.screen.fill((0, 0, 0))
            self.screen.blit(render_surface, (offset_x, offset_y))
            
            fps = self.clock.get_fps()
            mode_names = {
                InteractiveCamera.MODE_FPS: "FPS",
                InteractiveCamera.MODE_TRACKBALL: "Trackball",
                InteractiveCamera.MODE_ORBIT: "Orbit"
            }
            
            overlay_lines = [
                f"Viewer FPS: {fps:.1f}",
                f"Render: {self.render_width}x{self.render_height} [{self.render_resolution_label}]",
                f"Window: {self.window_width}x{self.window_height}",
                f"Mode: {mode_names[self.camera.mode]}",
            ]
            if self.camera.current_camera_idx >= 0:
                overlay_lines.append(f"Camera: {self.camera.current_camera_idx + 1}/{num_cams}")
            else:
                overlay_lines.append("Camera: Free")
            if self.sequence_manager:
                frame_info = f"Frame: {self.sequence_manager.current_frame + 1}/{self.sequence_manager.num_frames}"
                if self.sequence_manager.playing:
                    frame_info += " [▶ 播放中]"
                else:
                    frame_info += " [⏸ 暂停]"
                overlay_lines.append(frame_info)
                overlay_lines.append(f"Playback FPS: {self.sequence_manager.fps:.0f}")
                overlay_lines.append(self.sequence_manager.get_cache_status())
                
                # 显示缓存命中率
                stats = self.sequence_manager.get_load_stats()
                if stats['total_accesses'] > 0:
                    overlay_lines.append(f"Cache Hit: {stats['hit_rate']:.0f}%")
                
                if self.sequence_manager.sampled_point_count != self.sequence_manager.current_point_count:
                    overlay_lines.append(
                        f"Gaussians: {self.sequence_manager.sampled_point_count}/{self.sequence_manager.current_point_count}"
                    )
                else:
                    overlay_lines.append(f"Gaussians: {self.sequence_manager.current_point_count}")
            
            for idx, text in enumerate(overlay_lines):
                text_surface = font.render(text, True, (255, 255, 255))
                self.screen.blit(text_surface, (10, 10 + idx * 32))
            
            # 显示帮助信息（按H切换）
            if self.show_help:
                help_lines = [
                    "=== 快捷键帮助 (H关闭) ===",
                    "1/2/3/4: 720p/1080p/2K/4K",
                    "R: 循环切换分辨率",
                    "WASD/QE: 移动",
                    "IJKL/UO: 旋转",
                    "Y: Trackball模式",
                    "P/N: 切换相机",
                    "M: 截图",
                ]
                if self.sequence_manager:
                    help_lines.extend([
                        "空格: 播放/暂停",
                        "←/→: 上/下一帧",
                        "F: 切换播放速度",
                        "-/+: 调整FPS",
                    ])
                
                help_x = self.window_width - 220
                for idx, text in enumerate(help_lines):
                    text_surface = small_font.render(text, True, (200, 200, 200))
                    self.screen.blit(text_surface, (help_x, 10 + idx * 24))
            else:
                # 简单提示
                hint_surface = small_font.render("Press H for help", True, (150, 150, 150))
                self.screen.blit(hint_surface, (self.window_width - 140, 10))
            
            pygame.display.flip()
            self.clock.tick(60)
        
        if self.sequence_manager:
            self.sequence_manager.shutdown()
        pygame.quit()


class OpenCVViewer:
    """基于OpenCV的交互式查看器 (SIBR风格控制)"""
    
    def __init__(
        self,
        renderer,
        camera,
        sequence_manager=None,
        title="3DGS Viewer",
        window_size=None,
        auto_fit_window=False,
    ):
        self.renderer = renderer
        self.camera = camera
        self.render_width = camera.width
        self.render_height = camera.height
        self.window_width, self.window_height = window_size or (camera.width, camera.height)
        self.title = title
        self.sequence_manager = sequence_manager
        self.point_cloud = renderer.pc
        self.render_resolution_label = f"{self.render_width}x{self.render_height}"
        
        self.running = True
        self.left_mouse_pressed = False
        self.right_mouse_pressed = False
        self.last_mouse_pos = None
        self.screenshot_count = 0
        self.trackball_ratio = 0.75
    
    def set_render_resolution(self, width, height, label=None):
        self.render_width = max(64, int(width))
        self.render_height = max(64, int(height))
        self.render_resolution_label = label or f"{self.render_width}x{self.render_height}"
        self.camera.resize(self.render_width, self.render_height)
        print(f"渲染分辨率切换为: {self.render_resolution_label} ({self.render_width}x{self.render_height})")
    
    def reload_current_frame(self):
        if self.sequence_manager:
            frame_data = self.sequence_manager.get_current_frame_data()
            self.point_cloud.apply_frame(frame_data)
    
    def adjust_playback_fps(self, delta):
        if not self.sequence_manager:
            return
        new_fps = self.sequence_manager.adjust_fps(delta)
        print(f"播放FPS调整为: {new_fps:.1f}")
        
    def is_in_trackball_center(self, pos):
        """检查鼠标是否在Trackball中心区域"""
        cx, cy = self.window_width / 2, self.window_height / 2
        radius = min(self.window_width, self.window_height) / 2 * self.trackball_ratio
        dx = pos[0] - cx
        dy = pos[1] - cy
        return (dx * dx + dy * dy) < (radius * radius)
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_mouse_pressed = True
            self.last_mouse_pos = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.right_mouse_pressed = True
            self.last_mouse_pos = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.left_mouse_pressed = False
            if not self.right_mouse_pressed:
                self.last_mouse_pos = None
        elif event == cv2.EVENT_RBUTTONUP:
            self.right_mouse_pressed = False
            if not self.left_mouse_pressed:
                self.last_mouse_pos = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.last_mouse_pos:
                dx = x - self.last_mouse_pos[0]
                dy = y - self.last_mouse_pos[1]
                
                if self.camera.mode == InteractiveCamera.MODE_TRACKBALL:
                    in_center = self.is_in_trackball_center(self.last_mouse_pos)
                    if self.left_mouse_pressed:
                        if in_center:
                            self.camera.trackball_rotate(dx, dy)
                        else:
                            self.camera.trackball_roll(dx)
                    elif self.right_mouse_pressed:
                        if in_center:
                            self.camera.trackball_pan(-dx, dy)
                        else:
                            self.camera.trackball_zoom(-dy * 0.5)
                else:
                    # FPS模式
                    if self.left_mouse_pressed:
                        self.camera.move_right(dx * 0.1)
                    if self.right_mouse_pressed:
                        self.camera.move_up(-dy * 0.1)
                
                self.last_mouse_pos = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.camera.zoom(1)
            else:
                self.camera.zoom(-1)
                
    def run(self):
        num_cams = len(self.camera.cameras_info)
        print("\n" + "="*50)
        print("  3DGS Python交互式查看器 (OpenCV后备模式)")
        print("="*50)
        print("\n=== FPS模式控制 ===")
        print("  W/A/S/D - 移动")
        print("  Q/E - 下降/上升")
        print("  I/J/K/L - 视角旋转")
        print("  U/O - 滚转")
        print("\n=== 相机控制 ===")
        print("  P - 跳转到最近相机")
        print("  N - 下一个相机")
        print("  Y - 切换Trackball模式")
        print("  1/2/3/4 - 切换 720p/1080p/2K/4K 渲染")
        if self.sequence_manager:
            print("  空格 - 播放/暂停")
            print("  -/+ - 调整播放FPS")
        print("  M - 截图")
        print("  ESC - 退出")
        print("="*50 + "\n")
        
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, self.window_width, self.window_height)
        cv2.setMouseCallback(self.title, self.mouse_callback)
        
        last_time = time.time()
        
        while self.running:
            if self.sequence_manager:
                self.sequence_manager.service_prefetch()
                due_frames = self.sequence_manager.consume_due_frames() if self.sequence_manager.playing else 0
                if due_frames > 0:
                    self.sequence_manager.set_frame(self.sequence_manager.current_frame + due_frames)
                    self.reload_current_frame()
            
            rendered = self.renderer.render(self.camera)
            
            image = rendered.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            current_time = time.time()
            fps = 1.0 / max(current_time - last_time, 0.001)
            last_time = current_time
            
            cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"Render: {self.render_width}x{self.render_height}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            mode_names = {
                InteractiveCamera.MODE_FPS: "FPS",
                InteractiveCamera.MODE_TRACKBALL: "Trackball",
                InteractiveCamera.MODE_ORBIT: "Orbit"
            }
            cv2.putText(image, f"Mode: {mode_names[self.camera.mode]}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if self.camera.current_camera_idx >= 0:
                cv2.putText(image, f"Camera: {self.camera.current_camera_idx + 1}/{num_cams}", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.putText(image, "Free Camera", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if self.sequence_manager:
                cv2.putText(image, f"Playback FPS: {self.sequence_manager.fps:.1f}", (10, 190),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(image, self.sequence_manager.get_cache_status(), (10, 230),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(self.title, image)
            
            key = cv2.waitKeyEx(1) if hasattr(cv2, "waitKeyEx") else cv2.waitKey(1)
            if key == 27:
                self.running = False
            elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
                label, (width, height) = RUNTIME_RESOLUTION_KEYS[key]
                self.set_render_resolution(width, height, label=label)
            elif key in (ord('-'), 45) and self.sequence_manager:
                self.adjust_playback_fps(-5.0)
            elif key in (ord('+'), ord('='), 43, 61) and self.sequence_manager:
                self.adjust_playback_fps(5.0)
            elif key == ord(' ') and self.sequence_manager:
                self.sequence_manager.toggle_play()
            elif key in (2424832, 81) and self.sequence_manager:
                self.sequence_manager.prev_frame()
                self.reload_current_frame()
            elif key in (2555904, 83) and self.sequence_manager:
                self.sequence_manager.next_frame()
                self.reload_current_frame()
            elif key in (2359296, 80) and self.sequence_manager:
                self.sequence_manager.set_frame(0)
                self.reload_current_frame()
            elif key in (2293760, 87) and self.sequence_manager:
                self.sequence_manager.set_frame(self.sequence_manager.num_frames - 1)
                self.reload_current_frame()
            elif key == ord('w'):
                self.camera.move_forward(1)
            elif key == ord('s'):
                self.camera.move_forward(-1)
            elif key == ord('a'):
                self.camera.move_right(-1)
            elif key == ord('d'):
                self.camera.move_right(1)
            elif key == ord('q'):
                self.camera.move_up(-1)
            elif key == ord('e'):
                self.camera.move_up(1)
            elif key == ord('i'):
                self.camera.rotate_pitch(1)
            elif key == ord('k'):
                self.camera.rotate_pitch(-1)
            elif key == ord('j'):
                self.camera.rotate_yaw(-1)
            elif key == ord('l'):
                self.camera.rotate_yaw(1)
            elif key == ord('u'):
                self.camera.rotate_roll(1)
            elif key == ord('o'):
                self.camera.rotate_roll(-1)
            elif key == ord('y'):
                self.camera.switch_mode(InteractiveCamera.MODE_TRACKBALL)
            elif key == ord('b'):
                self.camera.switch_mode(InteractiveCamera.MODE_ORBIT)
            elif key == ord('p'):
                self.camera.snap_to_nearest_camera()
            elif key == ord('n'):
                self.camera.next_camera()
            elif key == ord('m'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}_{self.screenshot_count}.png"
                cv2.imwrite(filename, image)
                print(f"截图已保存: {filename}")
                self.screenshot_count += 1
        
        if self.sequence_manager:
            self.sequence_manager.shutdown()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="3D Gaussian Splatting Interactive Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 直接指定PLY文件
  python mult-frame_free-resolution_visualization.py path/to/point_cloud.ply
  
  # 4K渲染 + 窗口自动适配屏幕
  python mult-frame_free-resolution_visualization.py data/1/window_000/per_frame_ply --render-resolution 4k --window-size auto
  
  # 序列流式播放，指定播放FPS和缓存大小
  python mult-frame_free-resolution_visualization.py data/1/window_000/per_frame_ply --playback-fps 15 --gpu-cache-size 2 --cpu-cache-size 2
"""
    )
    
    # 位置参数 - 直接指定PLY文件或序列目录
    parser.add_argument("input", type=str, nargs="?", default=None,
                        help="PLY文件路径、序列目录或模型目录 (可选)")
    parser.add_argument("--path", "-s", type=str, default=None,
                        help="数据集路径 (用于加载cameras.json)")
    parser.add_argument("--model-path", "-m", type=str, default=None,
                        help="模型路径")
    parser.add_argument("--iteration", type=int, default=None,
                        help="指定迭代次数")
    parser.add_argument("--load_images", action="store_true",
                        help="(兼容SIBR参数)")
    parser.add_argument("--ply_path", type=str, default=None,
                        help="直接指定PLY文件路径 (等同于位置参数)")
    parser.add_argument("--width", type=int, default=1280,
                        help="初始窗口宽度 (兼容旧参数, 默认1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="初始窗口高度 (兼容旧参数, 默认720)")
    parser.add_argument("--render-resolution", type=str, default=None,
                        help="渲染分辨率，支持 1080p/2k/4k/1920x1080")
    parser.add_argument("--window-size", type=str, default=None,
                        help="初始窗口大小，支持 auto/1080p/2k/1920x1080")
    parser.add_argument("--playback-fps", type=float, default=30.0,
                        help="序列播放FPS，默认30")
    parser.add_argument("--sh_degree", type=int, default=None,
                        help="球谐函数度数")
    parser.add_argument("--load-mode", type=str, default=SequenceManager.LOAD_MODE_AUTO,
                        choices=[
                            SequenceManager.LOAD_MODE_AUTO,
                            SequenceManager.LOAD_MODE_STREAM,
                            SequenceManager.LOAD_MODE_PRELOAD_CPU,
                            SequenceManager.LOAD_MODE_PRELOAD_GPU,
                        ],
                        help="序列加载模式: auto(默认) / stream / preload_cpu / preload_gpu")
    parser.add_argument("--gpu-cache-size", type=int, default=2,
                        help="流式模式下GPU缓存帧数，默认2")
    parser.add_argument("--cpu-cache-size", type=int, default=2,
                        help="流式模式下CPU缓存帧数，默认2")
    parser.add_argument("--prefetch-count", type=int, default=1,
                        help="流式模式下向前预取的帧数，默认1")
    parser.add_argument("--max-gaussians", type=int, default=None,
                        help="每帧最多加载多少个高斯，适合大规模预览")
    parser.add_argument("--no-pin-memory", action="store_true",
                        help="禁用CPU pinned memory预取")
    parser.add_argument("--white_background", "-w", action="store_true",
                        help="使用白色背景")
    parser.add_argument("--use_opencv", action="store_true",
                        help="强制使用OpenCV显示")
    
    args = parser.parse_args()
    
    ply_path = None
    model_path = None
    data_path = None
    sequence_manager = None
    sequence_dir = None  # 暂存序列目录路径
    config = {'sh_degree': 3, 'white_background': False}
    cameras = []
    
    # 处理输入参数 (位置参数优先)
    input_path = args.input or args.ply_path
    
    if input_path:
        # 检查是否是序列目录（包含frame_*.ply文件）
        if os.path.isdir(input_path):
            ply_files = [f for f in os.listdir(input_path) if f.startswith('frame_') and f.endswith('.ply')]
            if ply_files:
                # 这是一个序列目录
                print(f"检测到序列目录: {input_path}")
                sequence_dir = input_path
                ply_path = os.path.join(input_path, sorted(ply_files)[0])
            elif os.path.exists(os.path.join(input_path, "cfg_args")):
                # 这是模型目录
                model_path = input_path
                data_path = args.path if args.path else model_path
                config = parse_cfg_args(model_path)
                
                point_cloud_dir = os.path.join(model_path, "point_cloud")
                
                if args.iteration:
                    iteration_dir = f"iteration_{args.iteration}"
                else:
                    iteration_dir = find_largest_iteration(point_cloud_dir)
                    if iteration_dir is None:
                        print(f"错误: 在 {point_cloud_dir} 中找不到iteration目录")
                        sys.exit(1)
                
                ply_path = os.path.join(point_cloud_dir, iteration_dir, "point_cloud.ply")
        elif input_path.endswith('.ply'):
            # 直接指定PLY文件
            ply_path = input_path
            # 尝试从PLY路径推断model_path
            parent = os.path.dirname(os.path.dirname(os.path.dirname(input_path)))
            if os.path.exists(os.path.join(parent, "cfg_args")):
                model_path = parent
                config = parse_cfg_args(model_path)
        else:
            print(f"错误: 无法识别的输入 '{input_path}'")
            sys.exit(1)
    elif args.model_path:
        model_path = args.model_path
        data_path = args.path if args.path else model_path
        
        config = parse_cfg_args(model_path)
        
        point_cloud_dir = os.path.join(model_path, "point_cloud")
        
        if args.iteration:
            iteration_dir = f"iteration_{args.iteration}"
        else:
            iteration_dir = find_largest_iteration(point_cloud_dir)
            if iteration_dir is None:
                print(f"错误: 在 {point_cloud_dir} 中找不到iteration目录")
                sys.exit(1)
        
        ply_path = os.path.join(point_cloud_dir, iteration_dir, "point_cloud.ply")
    else:
        parser.print_help()
        print("\n错误: 请指定PLY文件、模型目录或 --model-path")
        sys.exit(1)
    
    print(f"使用模型: {ply_path}")
    
    if not os.path.exists(ply_path):
        print(f"错误: 找不到点云文件 '{ply_path}'")
        sys.exit(1)
    
    if data_path:
        cameras_json = os.path.join(data_path, "cameras.json")
        cameras = load_cameras_from_json(cameras_json)
        if cameras:
            print(f"加载了 {len(cameras)} 个相机视角")
    
    try:
        render_size = parse_resolution(args.render_resolution) if args.render_resolution else (args.width, args.height)
        if args.window_size is not None:
            parsed_window_size = parse_resolution(args.window_size, allow_auto=True)
            auto_fit_window = parsed_window_size is None
            window_size = parsed_window_size
        else:
            auto_fit_window = bool(args.render_resolution and args.width == 1280 and args.height == 720)
            window_size = None if auto_fit_window else (args.width, args.height)
    except ValueError as exc:
        print(f"错误: {exc}")
        sys.exit(1)
    
    render_width, render_height = render_size
    if window_size is None:
        window_size = (render_width, render_height)
    
    sh_degree = args.sh_degree if args.sh_degree is not None else config['sh_degree']
    
    # 如果是序列目录，现在创建SequenceManager（需要sh_degree）
    if sequence_dir:
        sequence_manager = SequenceManager(
            sequence_dir,
            sh_degree=sh_degree,
            playback_fps=args.playback_fps,
            load_mode=args.load_mode,
            gpu_cache_size=clamp_positive_int(args.gpu_cache_size, 2),
            cpu_cache_size=clamp_positive_int(args.cpu_cache_size, 2),
            prefetch_count=max(0, int(args.prefetch_count)),
            pin_memory=not args.no_pin_memory,
            max_gaussians=args.max_gaussians,
        )
    
    if args.white_background or config['white_background']:
        bg_color = [1, 1, 1]
    else:
        bg_color = [0, 0, 0]
    
    print("正在初始化...")
    print(f"  SH度数: {sh_degree}")
    print(f"  背景: {'白色' if bg_color == [1,1,1] else '黑色'}")
    print(f"  渲染分辨率: {render_width}x{render_height}")
    if auto_fit_window:
        print(f"  初始窗口: 自动适配屏幕")
    else:
        print(f"  初始窗口: {window_size[0]}x{window_size[1]}")
    
    # 如果是序列，使用第一帧的数据创建点云对象（避免重复加载）
    if sequence_manager:
        frame_data = sequence_manager.get_current_frame_data()
        pc = GaussianPointCloud.from_frame(frame_data)
    else:
        pc = GaussianPointCloud(
            ply_path,
            sh_degree=sh_degree,
            max_gaussians=args.max_gaussians,
        )
    
    camera = InteractiveCamera(
        render_width, render_height,
        cameras_info=cameras,
        scene_center=pc.scene_center,
        scene_extent=pc.scene_extent
    )
    
    if cameras:
        camera.set_camera(0)
    
    renderer = GaussianRenderer(pc, bg_color=bg_color)
    
    use_pygame = PYGAME_AVAILABLE and not args.use_opencv
    
    if sequence_manager:
        print(f"\n序列播放模式:")
        print(f"  总帧数: {sequence_manager.num_frames}")
        print(f"  播放FPS: {sequence_manager.fps:.1f}")
        print(f"  加载模式: {sequence_manager.load_mode}")
        print(f"  缓存状态: {sequence_manager.get_cache_status()}")
        print(f"\n控制:")
        print(f"  空格键 - 播放/暂停")
        print(f"  左/右箭头 - 上一帧/下一帧")
        print(f"  HOME - 跳到第一帧")
        print(f"  END - 跳到最后一帧")
        print(f"  -/+ - 调整播放FPS")
    
    if use_pygame:
        viewer = PygameViewer(
            renderer,
            camera,
            sequence_manager,
            window_size=window_size,
            auto_fit_window=auto_fit_window,
        )
    elif CV2_AVAILABLE:
        viewer = OpenCVViewer(
            renderer,
            camera,
            sequence_manager,
            window_size=window_size,
            auto_fit_window=auto_fit_window,
        )
    else:
        print("错误: 需要安装pygame或opencv-python")
        sys.exit(1)
    
    viewer.run()


if __name__ == "__main__":
    main()
