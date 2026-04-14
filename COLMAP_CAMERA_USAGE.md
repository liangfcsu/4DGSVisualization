# 使用 COLMAP 真实相机位姿查看场景

## 功能说明

现在 player.py 支持从 COLMAP sparse 重建数据中加载真实相机位姿，让你可以在训练数据的原始拍摄位置查看重建的 3D 场景。

## 使用方法

### 1. 准备 COLMAP 数据

确保你有 COLMAP sparse 重建目录，通常包含以下文件：
```
data/sparse/
├── cameras.bin (或 cameras.txt)
├── images.bin  (或 images.txt)
└── points3D.bin (可选)
```

### 2. 运行播放器

使用 `--sparse` 参数指定 COLMAP sparse 目录：

```bash
# 单帧 PLY + COLMAP 相机
python player.py point_cloud.ply --sparse data/sparse

# 序列播放 + COLMAP 相机
python player.py sequence_dir --sparse data/sparse

# 完整示例
python player.py output/scene/point_cloud/iteration_30000/point_cloud.ply \
    --sparse colmap/sparse/0 \
    --render-resolution 2k
```

### 3. 相机切换操作

加载成功后，你可以通过以下方式切换相机：

- **工具栏下拉菜单**：在顶部工具栏选择"相机位姿"下拉框，直接选择相机
- **N 键**：切换到下一个相机位姿
- **P 键**：跳转到离当前视角最近的真实相机位姿
- **R 键**：重置回自由视角

### 4. 快捷键总结

| 快捷键 | 功能 |
|--------|------|
| N | 下一个相机位姿 |
| P | 跳转到最近的相机位姿 |
| R | 重置到自由视角 |
| W/A/S/D | 自由移动（自由视角模式） |
| Y | 切换 Trackball 模式 |
| B | 切换 Orbit 模式 |

## 支持的格式

- ✅ COLMAP 二进制格式（.bin）
- ✅ COLMAP 文本格式（.txt）
- ✅ SIMPLE_PINHOLE 相机模型
- ✅ PINHOLE 相机模型

## 注意事项

1. **坐标系转换**：代码会自动处理 COLMAP 的 world-to-camera 格式到 3DGS 的 camera-to-world 格式的转换
2. **相机数量**：如果有很多相机，建议使用 P 键快速跳转到最近的位姿
3. **自由视角**：在下拉框中选择"自由视角"可以退出固定相机模式，继续自由漫游

## 示例场景

假设你有以下目录结构：
```
project/
├── output/
│   └── scene/
│       └── point_cloud/
│           └── iteration_30000/
│               └── point_cloud.ply
└── colmap/
    └── sparse/
        └── 0/
            ├── cameras.bin
            └── images.bin
```

运行命令：
```bash
python player.py output/scene/point_cloud/iteration_30000/point_cloud.ply \
    --sparse colmap/sparse/0 \
    --render-resolution 4k
```

这样你就可以在原始拍摄位置查看重建结果，并使用 N 键在不同视角间切换！
