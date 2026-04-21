# 4DGS Viewer - 4D Gaussian Splatting 交互式可视化工具

一个功能强大的4D高斯点云序列可视化和编辑工具，支持实时播放、交互式编辑、批量导出等功能。

## ✨ 主要特性

- 🎬 **序列播放** - 流畅播放4DGS序列，支持多种缓存模式
- 🎨 **多种渲染模式** - RGB/Gaussian/Ring三种显示模式
- 🎮 **交互式相机** - FPS/Trackball/Orbit三种控制模式
- ✂️ **点云编辑** - 框选、多边形选择、隐藏、删除等编辑功能
- 📤 **批量导出** - 一键导出整个序列的编辑结果
- 🔄 **实时训练可视化** - 支持训练过程实时可视化
- 📷 **相机位姿** - 加载和切换COLMAP相机位姿
- ⚡ **高性能缓存** - 智能GPU/CPU缓存策略

## 📋 系统要求

### 硬件要求

- **GPU**: NVIDIA GPU with CUDA support (推荐 RTX 3060 或更高)
- **显存**: 至少 6GB (推荐 8GB+)
- **内存**: 至少 16GB (推荐 32GB+ 用于大序列)
- **存储**: SSD 推荐（用于快速加载序列）

### 软件要求

- **操作系统**: Linux (Ubuntu 20.04+ 推荐) / Windows (WSL2)
- **Python**: 3.8 - 3.11
- **CUDA**: 11.8 或 12.1
- **PyTorch**: 2.0.0+

## 🚀 快速开始

### 方式一：从GitHub下载

```bash
# 1. 克隆仓库
git clone https://github.com/YOUR_USERNAME/4DGSVisualization.git
cd 4DGSVisualization

# 2. 初始化和更新子模块
git submodule update --init --recursive

# 3. 创建Conda环境
conda create -n 3dgs python=3.11 -y
conda activate 3dgs

# 3. 安装PyTorch (根据你的CUDA版本)
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. 安装依赖
pip install -r requirements.txt

# 5. 编译CUDA扩展
cd submodules/diff-gaussian-rasterization
pip install -e .
cd ../..

# 6. 运行测试
python player.py data/fram8/points --sparse data/fram8/0
```

### 方式二：完整环境配置

```bash
# 1. 安装系统依赖 (Ubuntu)
sudo apt-get update
sudo apt-get install -y build-essential cmake git

# 2. 创建环境并安装所有依赖
conda create -n 3dgs python=3.11 -y
conda activate 3dgs

# 3. 安装核心依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install PyQt5 numpy plyfile opencv-python pygame tqdm

# 4. 编译diff-gaussian-rasterization
cd submodules/diff-gaussian-rasterization
pip install -e .
```

## 📖 使用方法

### 1. 启动方式

**空白启动（推荐）**

```bash
conda activate 3dgs
python player.py
```

然后通过GUI菜单加载数据：

- 文件 → 加载点云序列 → 选择序列文件夹
- 文件 → 加载相机参数 → 选择COLMAP sparse/0文件夹

**命令行启动**

```bash
# 加载序列和相机参数
python player.py <序列目录> --sparse <sparse目录>

# 示例
python player.py data/fram8/points --sparse data/fram8/0
python player.py /path/to/global_per_frame_ply --sparse /path/to/sparse/0
```

**高级参数**

```bash
python player.py <序列> --sparse <sparse> \
    --render-resolution 2k \
    --gpu-cache-size 10 \
    --cpu-cache-size 30 \
    --playback-fps 30
```

### 2. 快捷键

#### 播放控制

- `空格` - 播放/暂停
- `←/→` - 上一帧/下一帧
- `Home/End` - 第一帧/最后一帧
- `-/+` - 减速/加速播放

#### 相机控制

- `W/A/S/D` - 前进/左移/后退/右移
- `Q/E` - 下降/上升
- `I/K/J/L` - 旋转视角
- `U/O` - 左滚/右滚
- `Y` - 切换到Trackball模式
- `B` - 切换到Orbit模式
- `N` - 下一个相机位姿
- `P` - 跳转到最近的相机位姿
- `R` - 重置相机

#### 显示控制

- `G` - 循环切换显示模式 (RGB/Gaussian/Ring)
- `1/2/3/4` - 切换分辨率 (720p/1080p/2K/4K)
- `M` - 截图
- `F/F11` - 全屏
- `Tab` - 隐藏/显示侧栏
- `H` - 隐藏/显示HUD

#### 编辑功能

- `V` - 切换选择模式
- `X` - 多边形选择模式
- `Ctrl+A` - 全选
- `Ctrl+Shift+A` - 清空选择
- `Shift+C` - 清除持久框
- `Ctrl+I` - 反选
- `Shift+H` - 隐藏选中
- `Shift+U` - 恢复隐藏
- `Delete` - 删除选中
- `Shift+Delete` - 删除未选中
- `Shift+R` - 恢复删除

### 3. 鼠标操作

#### FPS/Orbit模式

- **左键拖动** - 旋转视角
- **右键拖动** - 平移视角
- **滚轮** - 缩放

#### Trackball模式

- **左键(中心)** - 球面旋转
- **左键(边缘)** - 滚转旋转
- **右键(中心)** - 平面平移
- **右键(边缘)** - 前后移动
- **滚轮** - 缩放

#### 选择模式

- **左键拖动** - 框选点云
- **Shift + 拖动** - 添加到选择
- **Ctrl + 拖动** - 从选择中移除
- **Alt + 拖动** - 反选
- **右键** - 完成多边形选择

### 4. 批量导出

导出编辑后的序列：

1. 文件 → 导出序列 PLY
2. 选择输出目录
3. 等待导出完成

导出内容：

- 只导出可见的高斯点
- 保持原始文件名
- 包含完整的3DGS参数

### 5. 训练可视化

实时监控3DGS训练过程：

1. 启动播放器：`python player.py`
2. 左侧面板 → 训练控制 → 选择训练数据
3. 点击"开始训练"
4. 实时查看训练进度和结果

## 📁 数据格式

### 序列目录结构

```
global_per_frame_ply/
├── frame_0000.ply
├── frame_0001.ply
├── frame_0002.ply
└── ...
```

### COLMAP Sparse目录

```
sparse/0/
├── cameras.txt
├── images.txt
└── points3D.txt
```

### PLY文件格式

支持标准的3D Gaussian Splatting PLY格式：

- 位置: x, y, z
- 球谐系数: f_dc_*, f_rest_*
- 缩放: scale_0, scale_1, scale_2
- 旋转: rot_0, rot_1, rot_2, rot_3
- 不透明度: opacity

## 🔧 配置说明

### 缓存模式

**preload_cpu** (推荐，默认)

- 预加载所有帧到CPU内存
- 按需传输到GPU
- 适合：内存充足的场景

**cpu_cache**

- 动态缓存到CPU
- 适合：超大序列

**gpu_cache**

- 直接缓存到GPU
- 适合：小序列，显存充足

### 性能调优

**GPU缓存大小**

```bash
--gpu-cache-size 10  # 默认10帧
```

**CPU缓存大小**

```bash
--cpu-cache-size 30  # 默认30帧
```

**IO工作线程**

```bash
--io-workers 24  # 默认自动检测
```

## 🐛 故障排除

### 问题：程序闪退或CUDA错误

**解决方案：**

1. 检查CUDA版本是否匹配
2. 更新显卡驱动
3. 减小GPU缓存：`--gpu-cache-size 5`

### 问题：加载序列很慢

**解决方案：**

1. 使用SSD存储序列
2. 增加IO线程：`--io-workers 32`
3. 使用预加载模式：`--load-mode preload_cpu`

### 问题：播放卡顿

**解决方案：**

1. 降低分辨率（按1键切换到720p）
2. 增加CPU缓存：`--cpu-cache-size 50`
3. 使用更快的存储设备

### 问题：内存不足

**解决方案：**

1. 减小CPU缓存：`--cpu-cache-size 10`
2. 使用流式模式：`--load-mode stream`
3. 限制高斯数量：`--max-gaussians 1000000`

## 📊 性能参考


| 序列大小     | 帧数    | GPU缓存 | CPU缓存 | 播放FPS |
| ------------ | ------- | ------- | ------- | ------- |
| 小 (<1GB)    | 8-20    | 10      | 30      | 60+     |
| 中 (1-5GB)   | 50-100  | 10      | 30      | 30-60   |
| 大 (5-20GB)  | 100-500 | 5       | 20      | 15-30   |
| 超大 (>20GB) | 500+    | 3       | 10      | 10-20   |

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目基于MIT许可证开源。

## 🙏 致谢

本项目基于以下开源项目：

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- PyQt5
- PyTorch
