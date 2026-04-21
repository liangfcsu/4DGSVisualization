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

from plyfile import PlyData, PlyElement
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

SH_C0 = 0.28209479177387814


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


def qvec2rotmat(qvec):
    """四元数转旋转矩阵 (COLMAP 格式: w, x, y, z)"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def read_colmap_cameras_binary(path_to_model_file):
    """读取 COLMAP cameras.bin 文件"""
    import struct
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = struct.unpack("Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_properties = struct.unpack("iiQQ", fid.read(24))
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = struct.unpack("Q", fid.read(8))[0]
            params = struct.unpack("d" * num_params, fid.read(8 * num_params))
            cameras[camera_id] = {
                'id': camera_id,
                'model': model_id,
                'width': int(width),
                'height': int(height),
                'params': params
            }
    return cameras


def read_colmap_cameras_text(path_to_model_file):
    """读取 COLMAP cameras.txt 文件"""
    cameras = {}
    with open(path_to_model_file, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            camera_id = int(elems[0])
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = tuple(map(float, elems[4:]))
            cameras[camera_id] = {
                'id': camera_id,
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras


def read_colmap_images_binary(path_to_model_file):
    """读取 COLMAP images.bin 文件"""
    import struct
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = struct.unpack("Q", fid.read(8))[0]
        for _ in range(num_reg_images):
            binary_image_properties = struct.unpack("idddddddi", fid.read(64))
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = struct.unpack("c", fid.read(1))[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = struct.unpack("c", fid.read(1))[0]
            num_points2D = struct.unpack("Q", fid.read(8))[0]
            fid.read(24 * num_points2D)  # 跳过 2D 点
            
            images[image_id] = {
                'id': image_id,
                'qvec': qvec,
                'tvec': tvec,
                'camera_id': camera_id,
                'name': image_name
            }
    return images


def read_colmap_images_text(path_to_model_file):
    """读取 COLMAP images.txt 文件"""
    images = {}
    with open(path_to_model_file, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            image_id = int(elems[0])
            qvec = np.array(tuple(map(float, elems[1:5])))
            tvec = np.array(tuple(map(float, elems[5:8])))
            camera_id = int(elems[8])
            image_name = elems[9]
            
            images[image_id] = {
                'id': image_id,
                'qvec': qvec,
                'tvec': tvec,
                'camera_id': camera_id,
                'name': image_name
            }
            
            # 跳过下一行（2D 点）
            fid.readline()
    return images


def load_cameras_from_colmap(sparse_dir):
    """从 COLMAP sparse 数据加载相机参数
    
    Args:
        sparse_dir: COLMAP sparse 重建目录路径（包含 cameras.bin/txt 和 images.bin/txt）
        
    Returns:
        cameras: 相机参数列表
    """
    cameras = []
    
    # 尝试读取二进制格式
    cameras_file_bin = os.path.join(sparse_dir, "cameras.bin")
    images_file_bin = os.path.join(sparse_dir, "images.bin")
    cameras_file_txt = os.path.join(sparse_dir, "cameras.txt")
    images_file_txt = os.path.join(sparse_dir, "images.txt")
    
    if os.path.exists(cameras_file_bin) and os.path.exists(images_file_bin):
        print(f"从 COLMAP 二进制文件加载相机: {sparse_dir}")
        colmap_cameras = read_colmap_cameras_binary(cameras_file_bin)
        colmap_images = read_colmap_images_binary(images_file_bin)
    elif os.path.exists(cameras_file_txt) and os.path.exists(images_file_txt):
        print(f"从 COLMAP 文本文件加载相机: {sparse_dir}")
        colmap_cameras = read_colmap_cameras_text(cameras_file_txt)
        colmap_images = read_colmap_images_text(images_file_txt)
    else:
        print(f"警告: 未找到 COLMAP 相机文件在 {sparse_dir}")
        return cameras
    
    # 转换为统一格式
    for img_id, img_data in sorted(colmap_images.items()):
        cam_id = img_data['camera_id']
        if cam_id not in colmap_cameras:
            continue
            
        cam_data = colmap_cameras[cam_id]
        
        # 从四元数和平移向量计算相机位置和旋转
        qvec = img_data['qvec']
        tvec = img_data['tvec']
        
        # COLMAP 使用 world-to-camera 变换
        # R 将世界坐标转到相机坐标，t 是相机中心在世界坐标系中的位置（经过 R 变换后）
        R_w2c = qvec2rotmat(qvec)
        
        # 相机中心位置：C = -R^T * t
        camera_center = -R_w2c.T @ tvec
        
        # 3DGS 使用 camera-to-world 格式
        # 所以我们需要相机在世界坐标中的旋转矩阵 R_c2w = R_w2c^T
        R_c2w = R_w2c.T
        
        # 提取焦距参数（支持 SIMPLE_PINHOLE 和 PINHOLE 模型）
        params = cam_data['params']
        if isinstance(cam_data['model'], int):
            # 二进制格式：model_id
            # 0: SIMPLE_PINHOLE (f, cx, cy)
            # 1: PINHOLE (fx, fy, cx, cy)
            if cam_data['model'] == 0:  # SIMPLE_PINHOLE
                fx = fy = params[0]
            elif cam_data['model'] == 1:  # PINHOLE
                fx, fy = params[0], params[1]
            else:
                fx = fy = params[0] if len(params) > 0 else 1000.0
        else:
            # 文本格式：model 是字符串
            if cam_data['model'] == 'SIMPLE_PINHOLE':
                fx = fy = params[0]
            elif cam_data['model'] == 'PINHOLE':
                fx, fy = params[0], params[1]
            else:
                fx = fy = params[0] if len(params) > 0 else 1000.0
        
        cameras.append({
            'id': img_id,
            'name': img_data['name'],
            'width': cam_data['width'],
            'height': cam_data['height'],
            'position': camera_center.astype(np.float32),
            'rotation': R_c2w.astype(np.float32),
            'fx': float(fx),
            'fy': float(fy)
        })
    
    print(f"成功加载 {len(cameras)} 个相机位姿")
    return cameras


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


def get_available_ram_bytes():
    meminfo_path = "/proc/meminfo"
    try:
        with open(meminfo_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        pass
    return None


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
PLAYER_SEQUENCE_PACKED_CACHE_VERSION = 1
PLAYER_SEQUENCE_PACKED_META_NAME = "packed_meta.json"
PLAYER_SEQUENCE_PACKED_FILE_NAMES = {
    "xyz": "packed_xyz.bin",
    "features": "packed_features.bin",
    "scaling": "packed_scaling.bin",
    "rotation": "packed_rotation.bin",
    "opacity": "packed_opacity.bin",
}


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


def load_packed_sequence_cache_metadata(cache_dir):
    meta_path = os.path.join(cache_dir, PLAYER_SEQUENCE_PACKED_META_NAME)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _packed_cache_file_paths(cache_dir, meta=None):
    file_names = dict(PLAYER_SEQUENCE_PACKED_FILE_NAMES)
    if meta and isinstance(meta.get("file_names"), dict):
        file_names.update(meta["file_names"])
    return {key: os.path.join(cache_dir, name) for key, name in file_names.items()}


def is_packed_sequence_cache_compatible(meta, sequence_dir, frame_files, sh_degree, max_gaussians):
    if not meta:
        return False
    if int(meta.get("packed_version", -1)) != PLAYER_SEQUENCE_PACKED_CACHE_VERSION:
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

    tensor_counts = meta.get("tensor_counts")
    offsets = meta.get("offsets")
    point_counts = meta.get("point_counts")
    sampled_counts = meta.get("sampled_counts")
    scene_centers = meta.get("scene_centers")
    scene_extents = meta.get("scene_extents")
    source_paths = meta.get("source_paths")
    if not all(isinstance(v, list) and len(v) == len(frame_files) for v in (
        tensor_counts,
        offsets,
        point_counts,
        sampled_counts,
        scene_centers,
        scene_extents,
        source_paths,
    )):
        return False
    if int(meta.get("feature_coeff_count", 0)) <= 0:
        return False
    if int(meta.get("total_points", 0)) < sum(int(v) for v in tensor_counts):
        return False
    return True


@lru_cache(maxsize=32)
def _open_packed_cache_memmap(path, shape):
    return np.memmap(path, mode='r', dtype=np.float32, shape=shape)


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
            build_packed_sequence_cache(
                sequence_dir,
                sh_degree=sh_degree,
                max_gaussians=max_gaussians,
                cache_dir=resolved_cache_dir,
                overwrite=False,
                verbose=verbose,
            )
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

    build_packed_sequence_cache(
        sequence_dir,
        sh_degree=sh_degree,
        max_gaussians=max_gaussians,
        cache_dir=resolved_cache_dir,
        overwrite=overwrite,
        verbose=verbose,
    )

    if verbose:
        print(f"播放器缓存构建完成，用时 {time.time() - start_time:.1f}s")
    return resolved_cache_dir


def build_packed_sequence_cache(
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

    packed_meta = load_packed_sequence_cache_metadata(resolved_cache_dir)
    packed_paths = _packed_cache_file_paths(resolved_cache_dir, packed_meta)
    if (
        not overwrite
        and is_packed_sequence_cache_compatible(packed_meta, sequence_dir, frame_files, sh_degree, max_gaussians)
        and all(os.path.exists(path) for path in packed_paths.values())
    ):
        if verbose:
            print(f"已存在可用顺序缓存: {resolved_cache_dir}")
        return resolved_cache_dir

    legacy_meta = load_sequence_cache_metadata(resolved_cache_dir)
    use_legacy_cache = is_sequence_cache_compatible(legacy_meta, sequence_dir, frame_files, sh_degree, max_gaussians)
    legacy_frame_paths = None
    if use_legacy_cache:
        legacy_frame_paths = [os.path.join(resolved_cache_dir, name) for name in legacy_meta["frame_cache_files"]]
        use_legacy_cache = all(os.path.exists(path) for path in legacy_frame_paths)

    total_frames = len(frame_files)
    start_time = time.time()
    if verbose:
        source_kind = "legacy-cache" if use_legacy_cache else "ply"
        print(f"开始构建顺序缓存: {resolved_cache_dir}")
        print(f"  帧数: {total_frames}, SH度数: {sh_degree}, max_gaussians: {max_gaussians}, 来源: {source_kind}")

    tmp_paths = {key: f"{path}.tmp" for key, path in _packed_cache_file_paths(resolved_cache_dir).items()}
    handles = {key: open(path, "wb") for key, path in tmp_paths.items()}

    offsets = []
    tensor_counts = []
    point_counts = []
    sampled_counts = []
    scene_centers = []
    scene_extents = []
    source_paths = []
    total_points = 0
    feature_coeff_count = None
    scaling_dim = None
    rotation_dim = None
    opacity_dim = None

    try:
        for idx, frame_file in enumerate(frame_files):
            if use_legacy_cache:
                frame = GaussianFrame.from_cache(
                    legacy_frame_paths[idx],
                    pin_memory=False,
                    verbose=verbose and idx == 0,
                )
            else:
                frame = GaussianFrame.from_ply(
                    os.path.join(sequence_dir, frame_file),
                    sh_degree=sh_degree,
                    pin_memory=False,
                    max_gaussians=max_gaussians,
                    verbose=verbose and idx == 0,
                )

            xyz_np = frame.xyz.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
            features_np = frame.features.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
            scaling_np = frame.scaling.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
            rotation_np = frame.rotation.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
            opacity_np = frame.opacity.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)

            if feature_coeff_count is None:
                feature_coeff_count = int(features_np.shape[1])
                scaling_dim = int(scaling_np.shape[1])
                rotation_dim = int(rotation_np.shape[1])
                opacity_dim = int(opacity_np.shape[1])

            count = int(xyz_np.shape[0])
            offsets.append(int(total_points))
            tensor_counts.append(count)
            point_counts.append(int(frame.point_count))
            sampled_counts.append(int(frame.sampled_count))
            scene_centers.append(frame.scene_center.astype(np.float32).tolist())
            scene_extents.append(float(frame.scene_extent))
            source_paths.append(frame.source_path)

            xyz_np.tofile(handles["xyz"])
            features_np.tofile(handles["features"])
            scaling_np.tofile(handles["scaling"])
            rotation_np.tofile(handles["rotation"])
            opacity_np.tofile(handles["opacity"])
            total_points += count

            if verbose and (idx == total_frames - 1 or (idx + 1) % max(1, min(20, total_frames)) == 0):
                print(f"  顺序缓存进度: {idx + 1}/{total_frames}")
    finally:
        for fh in handles.values():
            fh.close()

    meta_payload = {
        "packed_version": PLAYER_SEQUENCE_PACKED_CACHE_VERSION,
        "source_dir": os.path.abspath(sequence_dir),
        "frame_files": frame_files,
        "num_frames": total_frames,
        "sh_degree": int(sh_degree),
        "max_gaussians": None if max_gaussians is None else int(max_gaussians),
        "total_points": int(total_points),
        "tensor_counts": tensor_counts,
        "offsets": offsets,
        "point_counts": point_counts,
        "sampled_counts": sampled_counts,
        "scene_centers": scene_centers,
        "scene_extents": scene_extents,
        "source_paths": source_paths,
        "feature_coeff_count": int(feature_coeff_count or 0),
        "scaling_dim": int(scaling_dim or 0),
        "rotation_dim": int(rotation_dim or 0),
        "opacity_dim": int(opacity_dim or 0),
        "file_names": dict(PLAYER_SEQUENCE_PACKED_FILE_NAMES),
        "created_at": datetime.now().isoformat(),
    }

    for key, final_path in _packed_cache_file_paths(resolved_cache_dir, meta_payload).items():
        os.replace(tmp_paths[key], final_path)

    meta_tmp_path = os.path.join(resolved_cache_dir, f"{PLAYER_SEQUENCE_PACKED_META_NAME}.tmp")
    with open(meta_tmp_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, ensure_ascii=False, indent=2)
    os.replace(meta_tmp_path, os.path.join(resolved_cache_dir, PLAYER_SEQUENCE_PACKED_META_NAME))

    if verbose:
        print(f"顺序缓存构建完成，用时 {time.time() - start_time:.1f}s")
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
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except (TypeError, RuntimeError, ValueError):
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

    @classmethod
    def from_packed_cache(cls, cache_dir, packed_meta, frame_idx, pin_memory=False, verbose=False):
        frame_idx = int(frame_idx) % int(packed_meta["num_frames"])
        if verbose:
            print(f"正在加载顺序缓存帧: {frame_idx + 1}/{packed_meta['num_frames']}")

        offset = int(packed_meta["offsets"][frame_idx])
        count = int(packed_meta["tensor_counts"][frame_idx])
        total_points = int(packed_meta["total_points"])
        feature_coeff_count = int(packed_meta["feature_coeff_count"])
        scaling_dim = int(packed_meta["scaling_dim"])
        rotation_dim = int(packed_meta["rotation_dim"])
        opacity_dim = int(packed_meta["opacity_dim"])
        packed_paths = _packed_cache_file_paths(cache_dir, packed_meta)
        frame_slice = slice(offset, offset + count)

        xyz_np = np.array(
            _open_packed_cache_memmap(packed_paths["xyz"], (total_points, 3))[frame_slice],
            copy=True,
        )
        features_np = np.array(
            _open_packed_cache_memmap(packed_paths["features"], (total_points, feature_coeff_count, 3))[frame_slice],
            copy=True,
        )
        scaling_np = np.array(
            _open_packed_cache_memmap(packed_paths["scaling"], (total_points, scaling_dim))[frame_slice],
            copy=True,
        )
        rotation_np = np.array(
            _open_packed_cache_memmap(packed_paths["rotation"], (total_points, rotation_dim))[frame_slice],
            copy=True,
        )
        opacity_np = np.array(
            _open_packed_cache_memmap(packed_paths["opacity"], (total_points, opacity_dim))[frame_slice],
            copy=True,
        )

        scene_center = np.asarray(packed_meta["scene_centers"][frame_idx], dtype=np.float32)
        return cls(
            xyz=cls._maybe_pin(torch.from_numpy(xyz_np).contiguous(), pin_memory),
            features=cls._maybe_pin(torch.from_numpy(features_np).contiguous(), pin_memory),
            scaling=cls._maybe_pin(torch.from_numpy(scaling_np).contiguous(), pin_memory),
            rotation=cls._maybe_pin(torch.from_numpy(rotation_np).contiguous(), pin_memory),
            opacity=cls._maybe_pin(torch.from_numpy(opacity_np).contiguous(), pin_memory),
            sh_degree=int(packed_meta["sh_degree"]),
            scene_center=scene_center,
            scene_extent=float(packed_meta["scene_extents"][frame_idx]),
            source_path=packed_meta["source_paths"][frame_idx],
            point_count=int(packed_meta["point_counts"][frame_idx]),
            sampled_count=int(packed_meta["sampled_counts"][frame_idx]),
        )


class SequenceManager:
    """管理序列点云帧，默认使用流式 CPU/GPU 双缓存而非全量显存预加载"""
    
    LOAD_MODE_AUTO = "auto"
    LOAD_MODE_STREAM = "stream"
    LOAD_MODE_GPU_STREAM = "gpu_stream"
    LOAD_MODE_PRELOAD_CPU = "preload_cpu"
    LOAD_MODE_PRELOAD_GPU = "preload_gpu"
    AUTO_STREAM_MIN_BYTES = 12 * 1024 ** 3
    AUTO_STREAM_MIN_FRAMES = 96
    
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
        self.loop = True
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
        self.packed_cache_meta = None
        self.load_backend = "ply"
        self.background_preload_enabled = False
        self.background_preload_completed = False
        self.background_preload_queue = []
        self.background_preload_cursor = 0
        self.background_preload_reported = 0
        self.background_preload_report_step = 1
        
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
            packed_meta = load_packed_sequence_cache_metadata(resolved_cache_dir)
            packed_paths = _packed_cache_file_paths(resolved_cache_dir, packed_meta)
            has_packed_cache = (
                is_packed_sequence_cache_compatible(
                    packed_meta,
                    sequence_dir,
                    self.frame_files,
                    sh_degree,
                    max_gaussians,
                )
                and all(os.path.exists(path) for path in packed_paths.values())
            )
            if has_packed_cache:
                self.cache_dir = resolved_cache_dir
                self.packed_cache_meta = packed_meta
                self.load_backend = "packed"
                print(f"检测到顺序播放器缓存: {self.cache_dir}")

            meta = load_sequence_cache_metadata(resolved_cache_dir)
            has_compatible_cache = False
            if not has_packed_cache and is_sequence_cache_compatible(meta, sequence_dir, self.frame_files, sh_degree, max_gaussians):
                frame_cache_paths = [os.path.join(resolved_cache_dir, name) for name in meta["frame_cache_files"]]
                has_compatible_cache = all(os.path.exists(path) for path in frame_cache_paths)
                if has_compatible_cache:
                    self.cache_dir = resolved_cache_dir
                    self.cache_meta = meta
                    self.cache_frame_paths = frame_cache_paths
                    self.load_backend = "cache"
                    print(f"检测到播放器缓存: {self.cache_dir}")

            if not has_packed_cache and not has_compatible_cache:
                print(f"未找到可用播放器缓存，开始在序列目录内直接构建顺序缓存: {resolved_cache_dir}")
                build_packed_sequence_cache(
                    sequence_dir,
                    sh_degree=sh_degree,
                    max_gaussians=max_gaussians,
                    cache_dir=resolved_cache_dir,
                    overwrite=cache_overwrite,
                    verbose=True,
                )
            elif not has_packed_cache and has_compatible_cache:
                print(f"检测到旧版逐帧缓存，开始升级顺序缓存: {resolved_cache_dir}")
                build_packed_sequence_cache(
                    sequence_dir,
                    sh_degree=sh_degree,
                    max_gaussians=max_gaussians,
                    cache_dir=resolved_cache_dir,
                    overwrite=cache_overwrite,
                    verbose=True,
                )
            if not has_packed_cache:
                packed_meta = load_packed_sequence_cache_metadata(resolved_cache_dir)
                packed_paths = _packed_cache_file_paths(resolved_cache_dir, packed_meta)
                if (
                    is_packed_sequence_cache_compatible(
                        packed_meta,
                        sequence_dir,
                        self.frame_files,
                        sh_degree,
                        max_gaussians,
                    )
                    and all(os.path.exists(path) for path in packed_paths.values())
                ):
                    self.cache_dir = resolved_cache_dir
                    self.packed_cache_meta = packed_meta
                    self.load_backend = "packed"
                    print(f"顺序播放器缓存已就绪: {self.cache_dir}")
                else:
                    meta = load_sequence_cache_metadata(resolved_cache_dir)
                    if is_sequence_cache_compatible(meta, sequence_dir, self.frame_files, sh_degree, max_gaussians):
                        frame_cache_paths = [os.path.join(resolved_cache_dir, name) for name in meta["frame_cache_files"]]
                        if all(os.path.exists(path) for path in frame_cache_paths):
                            self.cache_dir = resolved_cache_dir
                            self.cache_meta = meta
                            self.cache_frame_paths = frame_cache_paths
                            self.load_backend = "cache"
                            print(f"播放器缓存已就绪: {self.cache_dir}")
        
        self._gpu_stream_mode = False  # 标记是否为 GPU Stream 模式
        self.load_mode = self._resolve_load_mode(load_mode)

        # GPU Stream 模式: 转化为优化参数的 stream 模式
        if self.load_mode == self.LOAD_MODE_GPU_STREAM:
            self._gpu_stream_mode = True
            self.gpu_cache_size = max(self.gpu_cache_size, 10)
            # CPU 缓存仅作中转暂存 (GPU窗口前方 3 帧)
            self._gpu_stream_staging = 3
            self.cpu_cache_size = self._gpu_stream_staging
            self.backward_prefetch = 0
            self.adaptive_prefetch = False
            self.prefetch_count = self.gpu_cache_size + self._gpu_stream_staging
            self.max_pending_cpu = max(self.prefetch_count, self.io_workers)
            # 内部使用 stream 基础设施
            self.load_mode = self.LOAD_MODE_STREAM

        if self.load_mode == self.LOAD_MODE_STREAM:
            self.backward_prefetch = 0
            self.adaptive_prefetch = False
            self.prefetch_count = max(self.prefetch_count, self.cpu_cache_size - 1)
            self.max_pending_cpu = max(self.max_pending_cpu, self.prefetch_count, self.io_workers)
        print(f"找到 {self.num_frames} 帧序列")
        print(f"序列磁盘占用: {self.total_sequence_bytes / (1024 ** 3):.2f} GB")
        
        if self.load_mode == self.LOAD_MODE_PRELOAD_GPU:
            self._preload_all_frames_to_gpu()
        elif self.load_mode == self.LOAD_MODE_PRELOAD_CPU:
            self._preload_all_frames_to_cpu()
        elif self._gpu_stream_mode:
            # ── GPU Stream 模式: 预加载前 N 帧到 GPU ──
            self.executor = ThreadPoolExecutor(
                max_workers=self.io_workers,
                thread_name_prefix="gpu-stream-loader",
            )
            gpu_preload_count = min(self.gpu_cache_size, self.num_frames)
            print(f"使用 GPU Stream 模式: GPU常驻 {self.gpu_cache_size} 帧滑动窗口 + CPU暂存 {self._gpu_stream_staging} 帧")
            print(
                f"  目标播放FPS: {self.fps:.1f}, IO线程: {self.io_workers}, 数据源: {self.load_backend}"
            )
            print(f"  正在预加载前 {gpu_preload_count} 帧到 GPU...")
            preload_start = time.time()
            for i in range(gpu_preload_count):
                frame = self._load_frame_cpu_sync(i, verbose=(i == 0))
                if i == 0:
                    self.scene_center = frame.scene_center.copy()
                    self.scene_extent = frame.scene_extent
                    self.active_sh_degree = frame.sh_degree
                    self.current_point_count = frame.point_count
                    self.sampled_point_count = frame.sampled_count
                frame_gpu = frame.to_device("cuda", non_blocking=False)
                self.gpu_cache[i] = frame_gpu
                if (i + 1) % 5 == 0 or i == gpu_preload_count - 1:
                    print(f"  GPU 预加载: {i + 1}/{gpu_preload_count}")
            preload_time = time.time() - preload_start
            print(f"  GPU 预加载完成: {len(self.gpu_cache)} 帧已在 GPU ({preload_time:.1f}s)")
            # 开始预取 GPU 窗口后方的暂存帧
            self.prefetch_around(0)
        else:
            self.executor = ThreadPoolExecutor(
                max_workers=self.io_workers,
                thread_name_prefix="ply-stream-loader",
            )
            print("使用流式滑动窗口模式: CPU前向窗口 + GPU热帧窗口 + 后台读盘")
            print(
                f"  目标播放FPS: {self.fps:.1f}, GPU缓存: {self.gpu_cache_size}, "
                f"CPU窗口: {self.cpu_cache_size}, 预读帧数: {self.prefetch_count}, "
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
        available_ram_bytes = get_available_ram_bytes()

        # 有 packed 缓存且序列较大 → 优先 GPU Stream 模式
        if (
            self.load_backend == "packed"
            and (
                self.total_sequence_bytes >= self.AUTO_STREAM_MIN_BYTES
                or self.num_frames >= self.AUTO_STREAM_MIN_FRAMES
            )
        ):
            print(
                "自动选择 gpu_stream 模式: 已有顺序缓存且序列较大，GPU常驻滑动窗口"
                f" (序列 {self.total_sequence_bytes / (1024 ** 3):.2f} GB, 帧数 {self.num_frames})"
            )
            return self.LOAD_MODE_GPU_STREAM
        
        if available_ram_bytes is not None:
            safe_available_budget = int(available_ram_bytes * 0.7)
            if self.total_sequence_bytes <= safe_available_budget:
                print(
                    "自动选择 preload_cpu 模式: 当前可用内存充足，优先避免播放期读盘抖动"
                    f" (序列 {self.total_sequence_bytes / (1024 ** 3):.2f} GB, "
                    f"可用内存 {available_ram_bytes / (1024 ** 3):.2f} GB)"
                )
                return self.LOAD_MODE_PRELOAD_CPU
        if total_ram_bytes is not None and self.total_sequence_bytes <= total_ram_bytes * 0.5:
            print(
                "自动选择 preload_cpu 模式: 序列尺寸在安全内存预算内"
                f" (序列 {self.total_sequence_bytes / (1024 ** 3):.2f} GB, "
                f"总内存 {total_ram_bytes / (1024 ** 3):.2f} GB)"
            )
            return self.LOAD_MODE_PRELOAD_CPU
        
        print("自动选择 gpu_stream 模式: 序列较大或无法判断内存容量，使用GPU流式缓存")
        return self.LOAD_MODE_GPU_STREAM
    
    def _frame_path(self, frame_idx):
        return self.frame_paths[self._normalize_frame_idx(frame_idx)]

    def _frame_cache_path(self, frame_idx):
        if self.cache_frame_paths is None:
            return None
        return self.cache_frame_paths[self._normalize_frame_idx(frame_idx)]

    def _stream_direction(self):
        return 1 if self.play_direction >= 0 else -1

    def _normalize_frame_idx(self, frame_idx, wrap=None):
        raw_idx = int(frame_idx)
        should_wrap = self.loop if wrap is None else bool(wrap)
        if should_wrap:
            return raw_idx % self.num_frames
        return max(0, min(raw_idx, self.num_frames - 1))

    def _remaining_steps_in_direction(self, direction=None):
        direction = self._stream_direction() if direction is None else (1 if direction >= 0 else -1)
        if self.loop:
            return self.num_frames - 1 if self.num_frames > 1 else 0
        if direction >= 0:
            return max(0, self.num_frames - 1 - self.current_frame)
        return max(0, self.current_frame)

    def at_playback_boundary(self, direction=None):
        return self._remaining_steps_in_direction(direction=direction) == 0

    def _stream_window_frames(self, total_frames):
        total_frames = max(1, int(total_frames))
        total_frames = min(self.num_frames, total_frames)
        direction = self._stream_direction()
        if self.loop:
            frames = []
            offset = 0
            while len(frames) < total_frames:
                frame_idx = (self.current_frame + direction * offset) % self.num_frames
                if frame_idx not in frames:
                    frames.append(frame_idx)
                offset += 1
            return frames

        if direction >= 0:
            end_idx = min(self.num_frames, self.current_frame + total_frames)
            return list(range(self.current_frame, end_idx))

        start_idx = max(0, self.current_frame - total_frames + 1)
        return list(range(self.current_frame, start_idx - 1, -1))

    def _cpu_window_frames(self):
        return self._stream_window_frames(self.cpu_cache_size)

    def _gpu_window_frames(self):
        return self._stream_window_frames(self.gpu_cache_size)

    def _cpu_prefetch_targets(self):
        return self._cpu_window_frames()[1:]

    def _gpu_prefetch_targets(self):
        return self._gpu_window_frames()[1:]

    def _prune_stream_windows(self):
        if self.load_mode != self.LOAD_MODE_STREAM:
            return

        cpu_keep = set(self._cpu_window_frames())
        gpu_keep = set(self._gpu_window_frames())

        for frame_idx in list(self.cpu_cache.keys()):
            if frame_idx not in cpu_keep:
                self.cpu_cache.pop(frame_idx, None)

        for frame_idx in list(self.gpu_cache.keys()):
            if frame_idx not in gpu_keep:
                self.gpu_cache.pop(frame_idx, None)

        for frame_idx, future in list(self.pending_cpu.items()):
            if frame_idx in cpu_keep:
                continue
            if future.cancel():
                self.pending_cpu.pop(frame_idx, None)
            elif future.done():
                try:
                    future.result()
                except Exception:
                    pass
                self.pending_cpu.pop(frame_idx, None)

        for frame_idx, pending_entry in list(self.pending_gpu.items()):
            if frame_idx in gpu_keep:
                continue
            _frame, _cpu_ref, ready_event = pending_entry
            if ready_event is None or ready_event.query():
                self.pending_gpu.pop(frame_idx, None)
    
    def _update_avg_stat(self, key, value, alpha=0.1):
        current = self.load_stats.get(key, 0.0)
        if current <= 0.0:
            self.load_stats[key] = float(value)
        else:
            self.load_stats[key] = alpha * float(value) + (1 - alpha) * current

    def _load_frame_cpu_sync(self, frame_idx, verbose=None):
        load_start = time.time()
        if self.packed_cache_meta is not None:
            frame = GaussianFrame.from_packed_cache(
                self.cache_dir,
                self.packed_cache_meta,
                frame_idx,
                pin_memory=self.pin_memory,
                verbose=self.verbose_frame_loads if verbose is None else verbose,
            )
        elif self.cache_frame_paths is not None:
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
        if self.load_mode == self.LOAD_MODE_STREAM:
            self._prune_stream_windows()
        else:
            while len(self.cpu_cache) > self.cpu_cache_size:
                self.cpu_cache.popitem(last=False)
    
    def _insert_gpu_cache(self, frame_idx, frame):
        self.gpu_cache[frame_idx] = frame
        self.gpu_cache.move_to_end(frame_idx)
        if self.load_mode == self.LOAD_MODE_STREAM:
            self._prune_stream_windows()
        else:
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
        print("正在准备首帧，并在后台预加载剩余帧到CPU内存...")
        self.cpu_cache_size = max(self.cpu_cache_size, self.num_frames)
        first_frame = self._load_frame_cpu_sync(0, verbose=self.verbose_frame_loads)
        self._insert_cpu_cache(0, first_frame)
        self.scene_center = first_frame.scene_center.copy()
        self.scene_extent = first_frame.scene_extent
        self.active_sh_degree = first_frame.sh_degree
        self.current_point_count = first_frame.point_count
        self.sampled_point_count = first_frame.sampled_count
        self.background_preload_report_step = max(1, min(16, self.num_frames))
        self.background_preload_reported = 1
        print(f"  CPU预加载进度: 1/{self.num_frames}")

        if self.num_frames <= 1:
            self.background_preload_completed = True
            print("CPU预加载完成，播放时只做GPU换帧")
            return

        self.executor = ThreadPoolExecutor(
            max_workers=self.io_workers,
            thread_name_prefix="ply-preload-loader",
        )
        self.background_preload_enabled = True
        self.background_preload_completed = False
        self.background_preload_queue = list(range(1, self.num_frames))
        self.background_preload_cursor = 0
        self._fill_background_preload_queue()
        print(f"首帧已就绪，剩余 {self.num_frames - 1} 帧正在后台预加载")

    def _report_background_preload_progress(self, force=False):
        if self.load_mode != self.LOAD_MODE_PRELOAD_CPU:
            return
        loaded_count = min(self.num_frames, len(self.cpu_cache))
        if not force and loaded_count < self.num_frames:
            if loaded_count - self.background_preload_reported < self.background_preload_report_step:
                return
        self.background_preload_reported = loaded_count
        print(f"  CPU预加载进度: {loaded_count}/{self.num_frames}")

    def _complete_background_preload(self):
        if self.background_preload_completed:
            return
        self.background_preload_enabled = False
        self.background_preload_completed = True
        self._report_background_preload_progress(force=True)
        print("CPU预加载完成，播放时只做GPU换帧")
        if self.executor is not None:
            try:
                self.executor.shutdown(wait=False, cancel_futures=False)
            except TypeError:
                self.executor.shutdown(wait=False)
            self.executor = None

    def _fill_background_preload_queue(self):
        if (
            self.load_mode != self.LOAD_MODE_PRELOAD_CPU
            or not self.background_preload_enabled
            or self.executor is None
        ):
            return

        while (
            len(self.pending_cpu) < self.max_pending_cpu
            and self.background_preload_cursor < len(self.background_preload_queue)
        ):
            frame_idx = self.background_preload_queue[self.background_preload_cursor]
            self.background_preload_cursor += 1
            if frame_idx in self.cpu_cache or frame_idx in self.gpu_cache or frame_idx in self.pending_cpu:
                continue
            self.pending_cpu[frame_idx] = self.executor.submit(
                self._load_frame_cpu_sync,
                frame_idx,
                False,
            )
    
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
        frame_idx = self._normalize_frame_idx(frame_idx)
        if frame_idx in self.cpu_cache or frame_idx in self.gpu_cache or frame_idx in self.pending_cpu:
            return
        self.pending_cpu[frame_idx] = self.executor.submit(
            self._load_frame_cpu_sync,
            frame_idx,
            False,
        )

    def _effective_prefetch_count(self):
        if self.load_mode == self.LOAD_MODE_STREAM:
            return max(0, min(self.num_frames - 1, self.cpu_cache_size - 1))
        if self.num_frames <= 1:
            return 0
        return max(0, min(self.num_frames - 1, self.prefetch_count))

    def _get_prefetch_targets(self, center_idx):
        if self.load_mode == self.LOAD_MODE_STREAM:
            return self._cpu_prefetch_targets()
        forward_count = self._effective_prefetch_count()
        if forward_count <= 0:
            return []
        return [
            self._normalize_frame_idx(center_idx + offset)
            for offset in range(1, forward_count + 1)
        ]

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
        self._prune_stream_windows()
        self._drop_stale_prefetch(targets)

        for frame_idx in targets:
            if len(self.pending_cpu) >= self.max_pending_cpu:
                break
            self._schedule_cpu_prefetch(frame_idx)
        return targets
    
    def service_prefetch(self):
        cpu_keep = set(self._cpu_window_frames()) if self.load_mode == self.LOAD_MODE_STREAM else None
        for frame_idx, future in list(self.pending_cpu.items()):
            if future.done():
                try:
                    frame = future.result()
                except Exception as exc:
                    print(f"预取失败(frame {frame_idx}): {exc}")
                else:
                    if cpu_keep is None or frame_idx in cpu_keep:
                        self._insert_cpu_cache(frame_idx, frame)
                finally:
                    self.pending_cpu.pop(frame_idx, None)
        
        if self.load_mode == self.LOAD_MODE_PRELOAD_CPU and self.background_preload_enabled:
            self._report_background_preload_progress(force=False)
            self._fill_background_preload_queue()
            if len(self.cpu_cache) >= self.num_frames and not self.pending_cpu:
                self._complete_background_preload()
            return

        if self.load_mode != self.LOAD_MODE_STREAM:
            return

        self._prune_stream_windows()
        targets = self._fill_cpu_prefetch_queue()
        for frame_idx in self._gpu_prefetch_targets():
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
        frame_idx = self._normalize_frame_idx(frame_idx)
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
        frame_idx = self._normalize_frame_idx(frame_idx)
        self.request_frame(frame_idx, prefer_device=prefer_device)

        if prefer_device == "cpu":
            return frame_idx in self.cpu_cache

        if frame_idx in self.gpu_cache:
            return True
        return self._resolve_pending_gpu(frame_idx, block=False) is not None
    
    def _resolve_cpu_frame(self, frame_idx):
        frame_idx = self._normalize_frame_idx(frame_idx)
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
        frame_idx = self._normalize_frame_idx(frame_idx)
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
        self.current_frame = self._normalize_frame_idx(frame_idx)
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
        self.current_frame = self._normalize_frame_idx(self.current_frame + 1)
        self.prefetch_around(self.current_frame)

    def prev_frame(self):
        self.play_direction = -1
        self.current_frame = self._normalize_frame_idx(self.current_frame - 1)
        self.prefetch_around(self.current_frame)

    def set_frame(self, frame_idx, update_direction=True):
        raw_idx = int(frame_idx)
        next_idx = self._normalize_frame_idx(raw_idx)
        if update_direction and raw_idx != self.current_frame:
            self.play_direction = 1 if raw_idx > self.current_frame else -1
        self.current_frame = next_idx
        self.prefetch_around(self.current_frame)
    
    def get_cache_status(self):
        if self.load_mode == self.LOAD_MODE_STREAM:
            if self._gpu_stream_mode:
                # GPU Stream 模式: 显示 GPU 窗口范围
                gpu_frames = self._gpu_window_frames()
                if gpu_frames:
                    start_idx = gpu_frames[0] + 1
                    end_idx = gpu_frames[-1] + 1
                    window_desc = f"GPU窗口:{start_idx}->{end_idx}"
                else:
                    window_desc = "GPU窗口:-"
                return (
                    f"gpu_stream/{self.load_backend} | GPU {len(self.gpu_cache)}/{self.gpu_cache_size} | "
                    f"暂存 {len(self.cpu_cache)} (+{len(self.pending_cpu)}) | "
                    f"{window_desc} | IO x{self.io_workers}"
                )
            window_frames = self._cpu_window_frames()
            if window_frames:
                start_idx = window_frames[0] + 1
                end_idx = window_frames[-1] + 1
                window_desc = f"窗口:{start_idx}->{end_idx}"
            else:
                window_desc = "窗口:-"
        elif self.load_mode == self.LOAD_MODE_PRELOAD_CPU and not self.background_preload_completed:
            window_desc = f"预加载:{len(self.cpu_cache)}/{self.num_frames}"
        else:
            window_desc = "窗口:all"
        return (
            f"{self.load_mode}/{self.load_backend} | GPU {len(self.gpu_cache)}/{self.gpu_cache_size} | "
            f"CPU {len(self.cpu_cache)}/{self.cpu_cache_size} (+{len(self.pending_cpu)}) | "
            f"{window_desc} | IO x{self.io_workers}"
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

    STATE_SELECTED = np.uint8(1)
    STATE_HIDDEN = np.uint8(2)
    STATE_DELETED = np.uint8(4)
    VOXEL_HASH_P1 = np.int64(73856093)
    VOXEL_HASH_P2 = np.int64(19349663)
    VOXEL_HASH_P3 = np.int64(83492791)

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
        self._masked_opacity_buffer = None
        self._frame_ref = None
        self._xyz = None
        self._features = None
        self._scaling = None
        self._rotation = None
        self._opacity = None
        self._source_opacity = None
        self._point_count = 0
        self._buffer_capacity = 0
        self._sample_index_cache = {}
        self._edit_state = None
        self._edit_state_key = None
        self._edit_state_store = {}
        self._deleted_voxel_hashes = np.empty((0,), dtype=np.int64)
        self._deleted_voxel_size = None
        self._deleted_voxel_hashes_torch = None
        self._deleted_voxel_hashes_torch_device = None
        self._cached_visible_count = 0
        self._cached_deleted_count = 0
        self._edit_state_version = 0

    def _buffer_device(self):
        return torch.device(self.device)

    def _copy_into(self, dst, src):
        non_blocking = False
        if src.device.type == "cuda":
            non_blocking = True
        elif src.device.type == "cpu" and src.is_pinned():
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
        self._masked_opacity_buffer = torch.empty((capacity, opacity_dim), dtype=torch.float32, device=device)
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
            or self._masked_opacity_buffer.shape[1:] != opacity_shape
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
        if device.type != "cpu":
            indices = indices.to(device=device, non_blocking=True)
        self._sample_index_cache[key] = indices
        return indices

    def _derive_edit_state_key(self, source_path):
        source_path = os.path.abspath(source_path) if source_path else ""
        base_name = os.path.basename(source_path)
        window_match = re.search(r"(window_\d+)", base_name)
        if window_match:
            bucket = window_match.group(1)
        else:
            bucket = "__global__"
        return f"{os.path.dirname(source_path)}::{bucket}"

    def _ensure_edit_state(self, point_count, source_path=None):
        point_count = int(point_count)
        state_key = self._derive_edit_state_key(source_path or self.ply_path)
        prev_key = self._edit_state_key

        state = self._edit_state_store.get(state_key)
        if state is None or state.shape[0] != point_count:
            state = np.zeros((max(0, point_count),), dtype=np.uint8)
            self._edit_state_store[state_key] = state
            self._edit_state_version += 1

        self._edit_state_key = state_key
        self._edit_state = state
        if prev_key != state_key:
            self._edit_state_version += 1

    def _visible_mask(self):
        if self._point_count <= 0:
            return np.zeros((0,), dtype=bool)
        if self._edit_state is None or self._edit_state.shape[0] != self._point_count:
            masked = np.zeros((self._point_count,), dtype=bool)
        else:
            masked = (self._edit_state & (self.STATE_HIDDEN | self.STATE_DELETED)) != 0
        if self._deleted_voxel_hashes.size == 0:
            return ~masked
        transfer_deleted = self._deleted_transfer_mask_torch(device=torch.device("cpu"))
        if transfer_deleted.numel() == 0:
            return ~masked
        return ~(masked | transfer_deleted.detach().cpu().numpy())

    def _ensure_masked_opacity_buffer(self):
        if self._source_opacity is None or self._point_count <= 0:
            return
        required_shape = (self._point_count, *self._source_opacity.shape[1:])
        if (
            self._masked_opacity_buffer is None
            or self._masked_opacity_buffer.shape[0] < required_shape[0]
            or self._masked_opacity_buffer.shape[1:] != required_shape[1:]
            or self._masked_opacity_buffer.device != self._source_opacity.device
        ):
            self._masked_opacity_buffer = torch.empty(
                required_shape,
                dtype=torch.float32,
                device=self._source_opacity.device,
            )

    def _selected_mask(self):
        if self._edit_state is None:
            return np.zeros((0,), dtype=bool)
        return (self._edit_state & self.STATE_SELECTED) != 0

    def _hash_voxel_coords_np(self, coords):
        coords = np.asarray(coords, dtype=np.int64)
        return (
            coords[:, 0] * self.VOXEL_HASH_P1
            ^ coords[:, 1] * self.VOXEL_HASH_P2
            ^ coords[:, 2] * self.VOXEL_HASH_P3
        ).astype(np.int64, copy=False)

    def _hash_voxel_coords_torch(self, coords):
        coords = coords.to(dtype=torch.int64)
        return (
            coords[:, 0] * int(self.VOXEL_HASH_P1)
            ^ coords[:, 1] * int(self.VOXEL_HASH_P2)
            ^ coords[:, 2] * int(self.VOXEL_HASH_P3)
        )

    def _resolve_deleted_voxel_size(self, scales_np=None):
        floor_size = max(float(self.scene_extent) * 0.001, 1e-4)
        if self._deleted_voxel_size is not None:
            return float(self._deleted_voxel_size)
        if scales_np is None or scales_np.size == 0:
            self._deleted_voxel_size = floor_size
            return self._deleted_voxel_size
        max_scale = np.max(scales_np, axis=1)
        median_scale = float(np.median(max_scale)) if max_scale.size > 0 else floor_size
        self._deleted_voxel_size = max(floor_size, median_scale * 1.5)
        return self._deleted_voxel_size

    def _deleted_transfer_mask_torch(self, indices=None, device=None):
        if self._xyz is None or self._point_count <= 0 or self._deleted_voxel_hashes.size == 0:
            if indices is None:
                length = self._point_count
            elif isinstance(indices, torch.Tensor):
                length = int(indices.numel())
            else:
                length = int(len(indices))
            target_device = device or self._buffer_device()
            return torch.zeros((length,), dtype=torch.bool, device=target_device)

        if device is None:
            device = self._xyz.device

        if (
            self._deleted_voxel_hashes_torch is None
            or self._deleted_voxel_hashes_torch_device != device
        ):
            self._deleted_voxel_hashes_torch = torch.from_numpy(self._deleted_voxel_hashes)
            if str(device) != "cpu":
                self._deleted_voxel_hashes_torch = self._deleted_voxel_hashes_torch.to(device=device, non_blocking=True)
            self._deleted_voxel_hashes_torch_device = device

        if indices is None:
            xyz = self._xyz
        else:
            gather_idx = indices.to(device=self._xyz.device, dtype=torch.long) if isinstance(indices, torch.Tensor) else torch.as_tensor(indices, device=self._xyz.device, dtype=torch.long)
            if gather_idx.numel() == 0:
                return torch.zeros((0,), dtype=torch.bool, device=device)
            xyz = self._xyz.index_select(0, gather_idx)

        coords = torch.round(xyz / float(self._deleted_voxel_size)).to(dtype=torch.int64)
        hashes = self._hash_voxel_coords_torch(coords)
        if hashes.device != device:
            hashes = hashes.to(device=device)
        return torch.isin(hashes, self._deleted_voxel_hashes_torch)

    def get_visible_mask_torch(self, indices=None, device=None):
        if device is None:
            device = self._buffer_device()
        if indices is None:
            length = self._point_count
        elif isinstance(indices, torch.Tensor):
            length = int(indices.numel())
        else:
            length = int(len(indices))
        if length <= 0:
            return torch.zeros((0,), dtype=torch.bool, device=device)

        if self._edit_state is None or self._edit_state.shape[0] != self._point_count:
            hidden_mask = torch.zeros((length,), dtype=torch.bool, device=device)
            local_deleted_mask = torch.zeros((length,), dtype=torch.bool, device=device)
        else:
            state = torch.from_numpy(self._edit_state)
            if str(device) != "cpu":
                state = state.to(device=device, non_blocking=True)
            if indices is not None:
                gather_idx = indices.to(device=device, dtype=torch.long) if isinstance(indices, torch.Tensor) else torch.as_tensor(indices, device=device, dtype=torch.long)
                state = state.index_select(0, gather_idx)
            hidden_mask = (state & self.STATE_HIDDEN) != 0
            local_deleted_mask = (state & self.STATE_DELETED) != 0
        transfer_deleted_mask = self._deleted_transfer_mask_torch(indices=indices, device=device)
        return ~(hidden_mask | local_deleted_mask | transfer_deleted_mask)

    def _accumulate_deleted_transfer_from_indices(self, indices):
        indices = self._sanitize_indices(indices)
        if indices.size == 0 or self._xyz is None or self._scaling is None:
            return 0

        xyz_np = self._xyz.index_select(
            0, torch.as_tensor(indices, device=self._xyz.device, dtype=torch.long)
        ).detach().cpu().numpy()
        scaling_np = self._scaling.index_select(
            0, torch.as_tensor(indices, device=self._scaling.device, dtype=torch.long)
        ).detach().cpu().numpy()

        voxel_size = self._resolve_deleted_voxel_size(scaling_np)
        coords = np.rint(xyz_np / voxel_size).astype(np.int64, copy=False)
        cell_radius = np.clip(
            np.ceil(np.max(scaling_np, axis=1) / voxel_size).astype(np.int64),
            0,
            1,
        )

        hashed_chunks = []
        unique_radii = np.unique(cell_radius)
        for radius in unique_radii:
            subset = coords[cell_radius == radius]
            if subset.size == 0:
                continue
            if radius <= 0:
                hashed_chunks.append(self._hash_voxel_coords_np(subset))
                continue
            offsets = np.array(
                [(dx, dy, dz) for dx in range(-radius, radius + 1)
                 for dy in range(-radius, radius + 1)
                 for dz in range(-radius, radius + 1)],
                dtype=np.int64,
            )
            expanded = (subset[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
            hashed_chunks.append(self._hash_voxel_coords_np(expanded))

        if not hashed_chunks:
            return 0

        new_hashes = np.unique(np.concatenate(hashed_chunks, axis=0))
        if self._deleted_voxel_hashes.size == 0:
            self._deleted_voxel_hashes = new_hashes
        else:
            self._deleted_voxel_hashes = np.unique(
                np.concatenate((self._deleted_voxel_hashes, new_hashes), axis=0)
            )
        self._deleted_voxel_hashes_torch = None
        self._deleted_voxel_hashes_torch_device = None
        return int(indices.size)

    def _refresh_edit_filters(self):
        if self._source_opacity is None or self._point_count <= 0:
            self._opacity = self._source_opacity
            self._cached_visible_count = 0
            self._cached_deleted_count = 0
            return

        device = self._source_opacity.device
        if self._edit_state is None or self._edit_state.shape[0] != self._point_count:
            hidden_t = torch.zeros((self._point_count,), dtype=torch.bool, device=device)
            local_deleted_t = torch.zeros((self._point_count,), dtype=torch.bool, device=device)
        else:
            hidden_mask = (self._edit_state & self.STATE_HIDDEN) != 0
            local_deleted_mask = (self._edit_state & self.STATE_DELETED) != 0
            hidden_t = torch.from_numpy(hidden_mask)
            local_deleted_t = torch.from_numpy(local_deleted_mask)
            if device.type != "cpu":
                hidden_t = hidden_t.to(device=device, non_blocking=True)
                local_deleted_t = local_deleted_t.to(device=device, non_blocking=True)

        transfer_deleted_t = self._deleted_transfer_mask_torch(device=device)
        masked_t = hidden_t | local_deleted_t | transfer_deleted_t

        self._cached_deleted_count = int((local_deleted_t | transfer_deleted_t).sum().item())
        self._cached_visible_count = int((~masked_t).sum().item())

        if not bool(masked_t.any().item()):
            self._opacity = self._source_opacity
            return

        self._ensure_masked_opacity_buffer()
        self._copy_into(self._masked_opacity_buffer[:self._point_count], self._source_opacity)
        visible_mask = (~masked_t).to(dtype=torch.float32)
        self._masked_opacity_buffer[:self._point_count].mul_(visible_mask[:, None])
        self._opacity = self._masked_opacity_buffer[:self._point_count]

    def _assign_active_frame(self, frame, point_count, active_sh_degree, src_xyz, src_features, src_scaling, src_rotation, src_opacity):
        self._point_count = point_count
        self._xyz = src_xyz
        self._features = src_features
        self._scaling = src_scaling
        self._rotation = src_rotation
        self._source_opacity = src_opacity
        self._opacity = src_opacity
        self.scene_center = frame.scene_center.copy()
        self.scene_extent = frame.scene_extent
        self.active_sh_degree = active_sh_degree
        self.max_sh_degree = frame.sh_degree
        self._ensure_edit_state(point_count, source_path=frame.source_path)
        self._refresh_edit_filters()

    def _sanitize_indices(self, indices):
        if self._edit_state is None or self._point_count <= 0:
            return np.empty((0,), dtype=np.int64)
        if isinstance(indices, torch.Tensor):
            arr = indices.detach().cpu().numpy()
        else:
            arr = np.asarray(indices)
        if arr.size == 0:
            return np.empty((0,), dtype=np.int64)
        arr = np.asarray(arr, dtype=np.int64).reshape(-1)
        arr = arr[(arr >= 0) & (arr < self._point_count)]
        if arr.size == 0:
            return arr
        arr = np.unique(arr)
        editable = (self._edit_state[arr] & (self.STATE_HIDDEN | self.STATE_DELETED)) == 0
        if np.any(editable) and self._deleted_voxel_hashes.size > 0:
            candidate = arr[editable]
            deleted_mask = self._deleted_transfer_mask_torch(indices=candidate).detach().cpu().numpy()
            editable_indices = np.flatnonzero(editable)
            editable[editable_indices] &= ~deleted_mask
        return arr[editable]

    def _mark_state_changed(self, refresh_visibility=False):
        self._edit_state_version += 1
        if refresh_visibility:
            self._refresh_edit_filters()

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
            self._assign_active_frame(
                frame,
                point_count,
                active_sh_degree,
                src_xyz,
                src_features,
                src_scaling,
                src_rotation,
                src_opacity,
            )
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

        self._assign_active_frame(
            frame,
            point_count,
            active_sh_degree,
            self._xyz_buffer[:point_count],
            self._features_buffer[:point_count],
            self._scaling_buffer[:point_count],
            self._rotation_buffer[:point_count],
            self._opacity_buffer[:point_count],
        )

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

    def export_current_ply(self, path):
        if self._xyz is None or self._features is None or self._scaling is None or self._rotation is None or self._source_opacity is None:
            raise RuntimeError("当前没有可导出的高斯点云数据")

        keep_mask = self._visible_mask()
        if keep_mask.size == 0 or not np.any(keep_mask):
            raise RuntimeError("当前没有可导出的可见高斯")

        xyz = self._xyz.detach().cpu().numpy()[keep_mask]
        features = self._features.detach().cpu().numpy()[keep_mask]
        scaling = self._scaling.detach().cpu().numpy()[keep_mask]
        rotation = self._rotation.detach().cpu().numpy()[keep_mask]
        opacity = self._source_opacity.detach().cpu().numpy()[keep_mask]

        opacity = np.clip(opacity, 1e-6, 1.0 - 1e-6)
        opacity_raw = np.log(opacity / (1.0 - opacity)).reshape(-1)
        scale_raw = np.log(np.clip(scaling, 1e-12, None))
        features_dc = features[:, 0, :]
        features_rest = np.transpose(features[:, 1:, :], (0, 2, 1)).reshape(features.shape[0], -1)

        dtype_fields = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ]
        dtype_fields.extend((f"f_rest_{i}", "f4") for i in range(features_rest.shape[1]))
        dtype_fields.append(("opacity", "f4"))
        dtype_fields.extend((f"scale_{i}", "f4") for i in range(scale_raw.shape[1]))
        dtype_fields.extend((f"rot_{i}", "f4") for i in range(rotation.shape[1]))

        vertex = np.empty(xyz.shape[0], dtype=dtype_fields)
        vertex["x"] = xyz[:, 0]
        vertex["y"] = xyz[:, 1]
        vertex["z"] = xyz[:, 2]
        vertex["nx"] = 0.0
        vertex["ny"] = 0.0
        vertex["nz"] = 0.0
        vertex["f_dc_0"] = features_dc[:, 0]
        vertex["f_dc_1"] = features_dc[:, 1]
        vertex["f_dc_2"] = features_dc[:, 2]
        for i in range(features_rest.shape[1]):
            vertex[f"f_rest_{i}"] = features_rest[:, i]
        vertex["opacity"] = opacity_raw
        for i in range(scale_raw.shape[1]):
            vertex[f"scale_{i}"] = scale_raw[:, i]
        for i in range(rotation.shape[1]):
            vertex[f"rot_{i}"] = rotation[:, i]

        ply = PlyData([PlyElement.describe(vertex, "vertex")], text=False)
        ply.write(path)

    def clear_selection(self):
        if self._edit_state is None:
            return 0
        selected = self._selected_mask()
        count = int(selected.sum())
        if count > 0:
            self._edit_state[selected] &= ~self.STATE_SELECTED
            self._mark_state_changed(refresh_visibility=False)
        return count

    def select_all(self):
        if self._edit_state is None:
            return 0
        visible = self._visible_mask()
        count = int(visible.sum())
        if count > 0:
            self._edit_state[visible] |= self.STATE_SELECTED
            self._mark_state_changed(refresh_visibility=False)
        return count

    def invert_selection(self):
        if self._edit_state is None:
            return 0
        visible = self._visible_mask()
        if not np.any(visible):
            return 0
        self._edit_state[visible] ^= self.STATE_SELECTED
        self._mark_state_changed(refresh_visibility=False)
        return int(self.selection_count)

    def apply_selection_indices(self, indices, op="set"):
        if self._edit_state is None:
            return 0
        indices = self._sanitize_indices(indices)
        visible = self._visible_mask()
        if op == "set":
            self._edit_state[visible] &= ~self.STATE_SELECTED
            if indices.size > 0:
                self._edit_state[indices] |= self.STATE_SELECTED
        elif op == "add":
            if indices.size > 0:
                self._edit_state[indices] |= self.STATE_SELECTED
        elif op == "remove":
            if indices.size > 0:
                self._edit_state[indices] &= ~self.STATE_SELECTED
        else:
            raise ValueError(f"Unsupported selection op: {op}")
        self._mark_state_changed(refresh_visibility=False)
        return int(self.selection_count)

    def hide_selected(self):
        if self._edit_state is None:
            return 0
        selected = self._selected_mask() & self._visible_mask()
        count = int(selected.sum())
        if count > 0:
            self._edit_state[selected] |= self.STATE_HIDDEN
            self._edit_state[selected] &= ~self.STATE_SELECTED
            self._mark_state_changed(refresh_visibility=True)
        return count

    def unhide_all(self):
        if self._edit_state is None:
            return 0
        hidden = (self._edit_state & self.STATE_HIDDEN) != 0
        count = int(hidden.sum())
        if count > 0:
            self._edit_state[hidden] &= ~self.STATE_HIDDEN
            self._mark_state_changed(refresh_visibility=True)
        return count

    def delete_selected(self):
        if self._edit_state is None:
            return 0
        selected = self._selected_mask() & self._visible_mask()
        count = int(selected.sum())
        if count > 0:
            self._accumulate_deleted_transfer_from_indices(np.flatnonzero(selected))
            self._edit_state[selected] |= self.STATE_DELETED
            self._edit_state[selected] &= ~self.STATE_SELECTED
            self._mark_state_changed(refresh_visibility=True)
        return count

    def restore_deleted(self):
        if self._edit_state is None:
            return 0
        count = self.deleted_count
        for state in self._edit_state_store.values():
            state &= ~self.STATE_DELETED
        if self._deleted_voxel_hashes.size > 0:
            self._deleted_voxel_hashes = np.empty((0,), dtype=np.int64)
            self._deleted_voxel_hashes_torch = None
            self._deleted_voxel_hashes_torch_device = None
        if self._deleted_voxel_size is not None:
            self._deleted_voxel_size = None
        if count > 0:
            self._mark_state_changed(refresh_visibility=True)
        return count

    def get_state_mask(self, device=None):
        if self._edit_state is None:
            return None
        tensor = torch.from_numpy(self._edit_state)
        if device is not None:
            tensor = tensor.to(device=device, non_blocking=(str(device) != "cpu"))
        return tensor

    def get_selected_indices(self):
        if self._edit_state is None:
            return np.empty((0,), dtype=np.int64)
        return np.flatnonzero(self._selected_mask()).astype(np.int64, copy=False)

    @property
    def selection_count(self):
        if self._edit_state is None:
            return 0
        return int(self._selected_mask().sum())

    @property
    def visible_count(self):
        return int(self._cached_visible_count)

    @property
    def hidden_count(self):
        if self._edit_state is None:
            return 0
        return int(((self._edit_state & self.STATE_HIDDEN) != 0).sum())

    @property
    def deleted_count(self):
        return int(self._cached_deleted_count)

    @property
    def edit_state_version(self):
        return int(self._edit_state_version)

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
    def get_source_opacity(self):
        return self._source_opacity if self._source_opacity is not None else self._opacity

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

    RENDER_MODE_SPLAT = "splat"
    RENDER_MODE_POINTS = "points"
    RENDER_MODE_RING = "ring"
    RENDER_MODE_LABELS = {
        RENDER_MODE_SPLAT: "高斯",
        RENDER_MODE_POINTS: "高斯点",
        RENDER_MODE_RING: "Ring模式",
    }
    
    def __init__(self, pc, bg_color=[0, 0, 0]):
        self.pc = pc
        self.bg_color = torch.tensor(bg_color, dtype=torch.float32, device=pc.get_xyz.device)
        self._means2d_buffer = None
        self._means2d_count = -1
        self.render_mode = self.RENDER_MODE_SPLAT
        self.point_size = 1.0
        self.point_opacity = 1.0
        self.ring_size = 0.3
        # 添加投影缓存以优化框选性能
        self._projection_cache = None
        self._projection_cache_key = None

    def _device(self):
        xyz = self.pc.get_xyz
        if xyz is not None:
            return xyz.device
        return self.bg_color.device

    def _background_color(self, device=None):
        if device is None:
            device = self._device()
        if self.bg_color.device != device:
            self.bg_color = self.bg_color.to(device=device)
        return self.bg_color

    def set_background_color(self, color):
        self.bg_color = torch.tensor(color, dtype=torch.float32, device=self._device())

    def set_render_mode(self, mode):
        mode = str(mode).lower()
        if mode not in self.RENDER_MODE_LABELS:
            raise ValueError(f"不支持的渲染模式: {mode}")
        self.render_mode = mode

    def cycle_render_mode(self):
        modes = [
            self.RENDER_MODE_SPLAT,
            self.RENDER_MODE_POINTS,
            self.RENDER_MODE_RING,
        ]
        next_idx = (modes.index(self.render_mode) + 1) % len(modes)
        self.render_mode = modes[next_idx]
        return self.render_mode

    def set_point_style(self, size=None, opacity=None):
        if size is not None:
            self.point_size = max(0.1, float(size))
        if opacity is not None:
            self.point_opacity = max(0.05, min(float(opacity), 1.0))

    def get_render_mode_label(self):
        return self.RENDER_MODE_LABELS.get(self.render_mode, self.render_mode)
    
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

    def _pick_scale_factor(self):
        if self.render_mode == self.RENDER_MODE_SPLAT:
            return 1.0
        return max(0.1, float(self.point_size))

    def _project_gaussians(self, camera, indices=None):
        xyz = self.pc.get_xyz
        scaling = self.pc.get_scaling
        if xyz is None or xyz.numel() == 0:
            return None

        device = xyz.device
        if indices is None:
            base_indices = torch.arange(xyz.shape[0], device=device, dtype=torch.long)
        else:
            if isinstance(indices, torch.Tensor):
                base_indices = indices.to(device=device, dtype=torch.long)
            else:
                base_indices = torch.as_tensor(indices, device=device, dtype=torch.long)
            if base_indices.numel() == 0:
                return None
            xyz = xyz.index_select(0, base_indices)
            scaling = scaling.index_select(0, base_indices)

        if indices is None:
            visible_mask = self.pc.get_visible_mask_torch(device=device)
        else:
            visible_mask = self.pc.get_visible_mask_torch(indices=base_indices, device=device)

        right = torch.as_tensor(camera.R[:, 0], dtype=torch.float32, device=device)
        down = torch.as_tensor(camera.R[:, 1], dtype=torch.float32, device=device)
        forward = torch.as_tensor(camera.R[:, 2], dtype=torch.float32, device=device)
        position = torch.as_tensor(camera.position, dtype=torch.float32, device=device)

        rel = xyz - position.unsqueeze(0)
        cam_x = rel @ right
        cam_y = rel @ down
        cam_z = rel @ forward

        tan_half_x = math.tan(camera.FoVx * 0.5)
        tan_half_y = math.tan(camera.FoVy * 0.5)
        focal_x = camera.width / max(1e-6, 2.0 * tan_half_x)
        focal_y = camera.height / max(1e-6, 2.0 * tan_half_y)

        safe_z = cam_z.clamp_min(max(camera.znear, 1e-4))
        screen_x = cam_x * (focal_x / safe_z) + camera.width * 0.5
        screen_y = cam_y * (focal_y / safe_z) + camera.height * 0.5

        scale_mul = self._pick_scale_factor()
        radii = torch.amax(scaling, dim=1) * max(focal_x, focal_y) * scale_mul * 2.5 / safe_z
        radii = radii.clamp(min=1.5, max=max(camera.width, camera.height) * 0.25)

        valid = visible_mask & (cam_z > max(camera.znear, 1e-4))
        valid &= screen_x >= -radii
        valid &= screen_x <= (camera.width + radii)
        valid &= screen_y >= -radii
        valid &= screen_y <= (camera.height + radii)

        return {
            "indices": base_indices,
            "screen_x": screen_x,
            "screen_y": screen_y,
            "depth": cam_z,
            "radius": radii,
            "valid": valid,
        }

    def _project_gaussian_centers(self, camera, indices=None):
        xyz = self.pc.get_xyz
        if xyz is None or xyz.numel() == 0:
            return None

        # 生成缓存键（仅在indices为None时缓存，即全量投影）
        if indices is None:
            cache_key = (
                id(camera),
                tuple(camera.position),
                tuple(camera.R.flatten().tolist()),
                camera.width,
                camera.height,
                camera.FoVx,
                camera.FoVy,
                self.pc.edit_state_version,  # 使用点云状态版本
            )
            # 检查缓存
            if self._projection_cache_key == cache_key:
                return self._projection_cache

        device = xyz.device
        if indices is None:
            base_indices = torch.arange(xyz.shape[0], device=device, dtype=torch.long)
        else:
            if isinstance(indices, torch.Tensor):
                base_indices = indices.to(device=device, dtype=torch.long)
            else:
                base_indices = torch.as_tensor(indices, device=device, dtype=torch.long)
            if base_indices.numel() == 0:
                return None
            xyz = xyz.index_select(0, base_indices)

        if indices is None:
            visible_mask = self.pc.get_visible_mask_torch(device=device)
        else:
            visible_mask = self.pc.get_visible_mask_torch(indices=base_indices, device=device)

        right = torch.as_tensor(camera.R[:, 0], dtype=torch.float32, device=device)
        down = torch.as_tensor(camera.R[:, 1], dtype=torch.float32, device=device)
        forward = torch.as_tensor(camera.R[:, 2], dtype=torch.float32, device=device)
        position = torch.as_tensor(camera.position, dtype=torch.float32, device=device)

        rel = xyz - position.unsqueeze(0)
        cam_x = rel @ right
        cam_y = rel @ down
        cam_z = rel @ forward

        tan_half_x = math.tan(camera.FoVx * 0.5)
        tan_half_y = math.tan(camera.FoVy * 0.5)
        focal_x = camera.width / max(1e-6, 2.0 * tan_half_x)
        focal_y = camera.height / max(1e-6, 2.0 * tan_half_y)

        safe_z = cam_z.clamp_min(max(camera.znear, 1e-4))
        screen_x = cam_x * (focal_x / safe_z) + camera.width * 0.5
        screen_y = cam_y * (focal_y / safe_z) + camera.height * 0.5

        valid = visible_mask & (cam_z > max(camera.znear, 1e-4))
        valid &= screen_x >= 0.0
        valid &= screen_x <= float(camera.width)
        valid &= screen_y >= 0.0
        valid &= screen_y <= float(camera.height)

        result = {
            "indices": base_indices,
            "screen_x": screen_x,
            "screen_y": screen_y,
            "depth": cam_z,
            "valid": valid,
        }
        
        # 缓存全量投影结果
        if indices is None:
            self._projection_cache = result
            self._projection_cache_key = cache_key
        
        return result

    def pick_point(self, camera, norm_x, norm_y, min_pick_radius=6.0, fallback_radius=14.0):
        with torch.inference_mode():
            projected = self._project_gaussians(camera)
            if projected is None:
                return None

            valid = projected["valid"]
            if not bool(valid.any().item()):
                return None

            px = float(norm_x) * camera.width
            py = float(norm_y) * camera.height
            dx = projected["screen_x"] - px
            dy = projected["screen_y"] - py
            dist2 = dx * dx + dy * dy
            eff_radius = torch.clamp(projected["radius"], min=float(min_pick_radius))

            hits = valid & (dist2 <= eff_radius * eff_radius)
            if not bool(hits.any().item()):
                hits = valid & (dist2 <= float(fallback_radius) ** 2)
                if not bool(hits.any().item()):
                    return None

            candidate_idx = torch.nonzero(hits, as_tuple=False).squeeze(1)
            if candidate_idx.numel() == 1:
                best = candidate_idx[0]
            else:
                depth = projected["depth"].index_select(0, candidate_idx)
                radius2 = eff_radius.index_select(0, candidate_idx).pow(2).clamp_min(1e-6)
                pick_score = depth / depth.max().clamp_min(1e-6)
                pick_score = pick_score + 0.15 * dist2.index_select(0, candidate_idx) / radius2
                best = candidate_idx[torch.argmin(pick_score)]
            return int(projected["indices"][best].item())

    def pick_rect(self, camera, start_norm, end_norm):
        with torch.inference_mode():
            projected = self._project_gaussian_centers(camera)
            if projected is None:
                return np.empty((0,), dtype=np.int64)

            x0 = min(float(start_norm[0]), float(end_norm[0])) * camera.width
            x1 = max(float(start_norm[0]), float(end_norm[0])) * camera.width
            y0 = min(float(start_norm[1]), float(end_norm[1])) * camera.height
            y1 = max(float(start_norm[1]), float(end_norm[1])) * camera.height

            hits = projected["valid"]
            hits &= projected["screen_x"] >= x0
            hits &= projected["screen_x"] <= x1
            hits &= projected["screen_y"] >= y0
            hits &= projected["screen_y"] <= y1
            if not bool(hits.any().item()):
                return np.empty((0,), dtype=np.int64)

            return (
                projected["indices"]
                .index_select(0, torch.nonzero(hits, as_tuple=False).squeeze(1))
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=False)
            )

    def get_selection_overlay(self, camera, max_points=384):
        selected = self.pc.get_selected_indices()
        if selected.size == 0:
            return []

        device = self._device()
        indices = torch.as_tensor(selected, device=device, dtype=torch.long)
        if indices.numel() > int(max_points):
            sample_ids = torch.linspace(
                0,
                indices.numel() - 1,
                steps=int(max_points),
                device=device,
            ).long()
            indices = indices.index_select(0, sample_ids)

        with torch.inference_mode():
            projected = self._project_gaussians(camera, indices=indices)
            if projected is None:
                return []
            valid = projected["valid"]
            if not bool(valid.any().item()):
                return []

            xs = projected["screen_x"][valid].detach().cpu().numpy()
            ys = projected["screen_y"][valid].detach().cpu().numpy()
            rs = projected["radius"][valid].clamp(3.0, 14.0).detach().cpu().numpy()
            return [
                (float(x), float(y), float(r))
                for x, y, r in zip(xs.tolist(), ys.tolist(), rs.tolist())
            ]
        
    def _render_gaussians(self, camera, resolution_scale=1.0):
        with torch.inference_mode():
            cam = camera.get_camera()
            scale = float(max(0.1, min(resolution_scale, 1.0)))
            image_width = max(1, int(round(cam.image_width * scale)))
            image_height = max(1, int(round(cam.image_height * scale)))
            
            screenspace_points = self._get_screenspace_buffer()
            bg = self._background_color(screenspace_points.device)
            
            kwargs = {
                'image_height': image_height,
                'image_width': image_width,
                'tanfovx': math.tan(cam.FoVx * 0.5),
                'tanfovy': math.tan(cam.FoVy * 0.5),
                'bg': bg,
                'scale_modifier': 1.0,
                'viewmatrix': cam.world_view_transform,
                'projmatrix': cam.full_proj_transform,
                'sh_degree': self.pc.active_sh_degree,
                'campos': cam.camera_center,
                'prefiltered': False,
                'debug': False,
            }
            if hasattr(GaussianRasterizationSettings, '_fields'):
                if 'antialiasing' in GaussianRasterizationSettings._fields:
                    kwargs['antialiasing'] = False
                if 'ring_mode' in GaussianRasterizationSettings._fields:
                    kwargs['ring_mode'] = False
                    kwargs['ring_size'] = 0.3
            
            raster_settings = GaussianRasterizationSettings(**kwargs)
            
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

    def _render_points(self, camera, resolution_scale=1.0):
        with torch.inference_mode():
            cam = camera.get_camera()
            scale = float(max(0.1, min(resolution_scale, 1.0)))
            image_width = max(1, int(round(cam.image_width * scale)))
            image_height = max(1, int(round(cam.image_height * scale)))
            device = self._device()
            xyz = self.pc.get_xyz
            if xyz is None or xyz.numel() == 0:
                return torch.zeros((3, image_height, image_width), dtype=torch.float32, device=device)

            screenspace_points = self._get_screenspace_buffer()
            bg = self._background_color(device)
            colors_precomp = torch.clamp(self.pc.get_features[:, 0, :] * SH_C0 + 0.5, 0.0, 1.0)
            opacities = (self.pc.get_opacity * self.point_opacity).clamp(0.0, 1.0)
            scales = self.pc.get_scaling * self.point_size

            kwargs = {
                'image_height': image_height,
                'image_width': image_width,
                'tanfovx': math.tan(cam.FoVx * 0.5),
                'tanfovy': math.tan(cam.FoVy * 0.5),
                'bg': bg,
                'scale_modifier': 1.0,
                'viewmatrix': cam.world_view_transform,
                'projmatrix': cam.full_proj_transform,
                'sh_degree': 0,
                'campos': cam.camera_center,
                'prefiltered': False,
                'debug': False,
            }
            if hasattr(GaussianRasterizationSettings, '_fields'):
                if 'antialiasing' in GaussianRasterizationSettings._fields:
                    kwargs['antialiasing'] = False
                if 'ring_mode' in GaussianRasterizationSettings._fields:
                    kwargs['ring_mode'] = False
                    kwargs['ring_size'] = 0.3
            raster_settings = GaussianRasterizationSettings(**kwargs)

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            result = rasterizer(
                means3D=xyz,
                means2D=screenspace_points,
                shs=None,
                colors_precomp=colors_precomp,
                opacities=opacities,
                scales=scales,
                rotations=self.pc.get_rotation,
                cov3D_precomp=None
            )

            if len(result) == 2:
                rendered_image, _radii = result
            else:
                rendered_image, _radii, _depth = result
            return rendered_image.clamp(0, 1)

    def _render_ring(self, camera, resolution_scale=1.0):
        """Ring 模式: 利用 CUDA rasterizer 原生 ring_mode 参数。
        
        在光栅化 fragment 阶段修改 alpha:
        - power = -0.5 * (conic 二次型) → 归一化距离 A = -power/4
        - 内部 (A < 1-ringSize): alpha = max(0.05, alpha) → 近乎透明
        - 环带 (A >= 1-ringSize): alpha = 0.6 → 清晰可见
        效果: 每个高斯显示为环形, 中心镂空, 边界清晰
        """
        with torch.inference_mode():
            cam = camera.get_camera()
            scale = float(max(0.1, min(resolution_scale, 1.0)))
            image_width = max(1, int(round(cam.image_width * scale)))
            image_height = max(1, int(round(cam.image_height * scale)))
            device = self._device()
            xyz = self.pc.get_xyz
            if xyz is None or xyz.numel() == 0:
                return torch.zeros((3, image_height, image_width), dtype=torch.float32, device=device)

            bg = self._background_color(device)
            screenspace = self._get_screenspace_buffer()

            # 使用训练好的原始不透明度, 保留视角相关的高斯贡献度
            opacities = self.pc.get_opacity
            scales = self.pc.get_scaling * self.point_size

            raster_kwargs = {
                'image_height': image_height,
                'image_width': image_width,
                'tanfovx': math.tan(cam.FoVx * 0.5),
                'tanfovy': math.tan(cam.FoVy * 0.5),
                'bg': bg,
                'scale_modifier': 1.0,
                'viewmatrix': cam.world_view_transform,
                'projmatrix': cam.full_proj_transform,
                'sh_degree': self.pc.active_sh_degree,
                'campos': cam.camera_center,
                'prefiltered': False,
                'debug': False,
                'ring_mode': True,
                'ring_size': getattr(self, 'ring_size', 0.3),
            }
            if hasattr(GaussianRasterizationSettings, '_fields'):
                if 'antialiasing' in GaussianRasterizationSettings._fields:
                    raster_kwargs['antialiasing'] = False

            raster_settings = GaussianRasterizationSettings(**raster_kwargs)
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            result = rasterizer(
                means3D=xyz,
                means2D=screenspace,
                shs=self.pc.get_features,
                colors_precomp=None,
                opacities=opacities,
                scales=scales,
                rotations=self.pc.get_rotation,
                cov3D_precomp=None,
            )
            rendered = result[0] if isinstance(result, tuple) else result
            return rendered.clamp(0, 1)

    def render(self, camera, resolution_scale=1.0):
        """渲染一帧"""
        if self.render_mode == self.RENDER_MODE_RING:
            return self._render_ring(camera, resolution_scale=resolution_scale)
        if self.render_mode == self.RENDER_MODE_SPLAT:
            return self._render_gaussians(camera, resolution_scale=resolution_scale)

        return self._render_points(camera, resolution_scale=resolution_scale)


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
            gpu_cache_size=clamp_positive_int(args.gpu_cache_size, 10),
            cpu_cache_size=clamp_positive_int(args.cpu_cache_size, 3),
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
