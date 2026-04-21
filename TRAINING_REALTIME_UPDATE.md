# 实时训练可视化 - 更新说明

## ✅ 已完成的改进

### 1. 训练输出解析优化

**改进内容：**
- 修改正则表达式以正确匹配 `train.py` 的 tqdm 进度条输出
- 支持解析：进度百分比、迭代次数、Loss值、Depth Loss、点数统计

**匹配格式：**
```
Training progress:  45%|████      | 13500/30000 [05:23<06:35, Loss=0.0234567, Depth Loss=0.0001234]
[ITER 7000] Saving Gaussians
Number of points at beginning of iteration: 123456
```

### 2. 右侧面板训练信息显示

**新增功能：**
- 在右侧信息面板添加 **TRAINING INFO** 卡片
- 实时显示：
  - 训练状态（训练中/已完成/已停止）
  - 当前迭代 / 总迭代（百分比）
  - Loss值（6位小数）
  - 高斯点数量

**显示效果：**
```
TRAINING INFO
状态      训练中
迭代      1,000 / 30,000 (3%)
Loss      0.012345
点数      50,000
```

### 3. 实时可视化增强

**实现方式：**
- 定期检查训练输出目录的 `point_cloud/iteration_*` 文件夹
- 检测到新的 checkpoint PLY 文件自动加载到渲染器
- 窗口标题实时显示当前训练迭代次数
- 每500次迭代显示toast提示

**自动加载流程：**
```
训练进行中
   ↓
每保存一次checkpoint (如 iteration_7000)
   ↓
自动检测新的 point_cloud.ply
   ↓
加载到渲染器并更新显示
   ↓
用户实时看到训练进度！
```

### 4. 状态更新优化

**改进：**
- 状态更新频率控制：每0.5秒更新一次（避免UI刷新过频繁）
- 双面板同步更新：左侧控制面板 + 右侧信息面板
- 训练完成/停止/出错时自动隐藏训练信息卡片

## 🎯 使用方法

### 启动训练

1. 打开左侧面板的 **"训练控制"** 部分
2. 选择训练数据路径（包含 `images/` 和 `sparse/` 的目录）
3. 选择输出目录
4. 设置迭代次数（默认30000）
5. 点击 **"▶ 开始训练"**

### 观察训练进度

**左侧面板显示：**
- 状态：训练中
- 进度：1000 / 30000
- Loss: 0.012345
- 点数: 50,000

**右侧面板显示：**
- TRAINING INFO 卡片自动出现
- 与左侧面板同步显示相同信息
- 格式更紧凑，适合实时监控

**播放窗口：**
- 实时显示训练中的高斯点云
- 每次保存checkpoint时自动加载新的PLY
- 窗口标题显示当前迭代次数：`[训练迭代: 7000]`

## 📊 实时可视化特性

### 自动加载时机

训练脚本会在以下迭代保存checkpoint：
- 默认：7000, 30000（最终）
- 可通过 `--save_iterations` 参数自定义

每次保存时：
1. 训练脚本输出：`[ITER 7000] Saving Gaussians`
2. 训练管理器检测到保存信号
3. 自动检查并加载新的 PLY 文件
4. 渲染器更新显示
5. 用户看到最新的训练结果

### 窗口标题显示

```
4DGS Viewer [训练迭代: 7000]  ← 实时更新迭代次数
```

### Toast 提示

每500次迭代显示一次：
```
训练进度: 7000 次迭代，点数: 123,456
```

## 🔧 技术细节

### 训练输出解析正则表达式

```python
# 进度条匹配
progress_pattern = re.compile(r'Training progress:\s+(\d+)%.*?(\d+)/(\d+)')

# Loss匹配（忽略大小写）
loss_pattern = re.compile(r'Loss[=:]\s*([\d.]+)', re.IGNORECASE)

# Depth Loss匹配
depth_loss_pattern = re.compile(r'Depth Loss[=:]\s*([\d.]+)', re.IGNORECASE)

# 保存检查点匹配
saving_pattern = re.compile(r'\[ITER (\d+)\] Saving')

# 点数匹配
points_pattern = re.compile(r'Number of points at beginning.*?(\d+)', re.IGNORECASE)
```

### 状态更新频率

```python
# 避免更新过于频繁导致UI卡顿
last_update_time = time.time()
if current_time - last_update_time >= 0.5:  # 每0.5秒更新一次
    # 更新UI
```

### PLY文件检测

```python
# 每20行输出检查一次新的PLY文件
check_counter += 1
if check_counter >= 20:
    check_counter = 0
    self._check_for_new_ply()
```

## 🎨 界面布局

```
┌─────────────┬──────────────────────┬─────────────┐
│ 左侧面板     │   播放窗口            │  右侧面板    │
│             │                      │             │
│ 训练控制     │   [实时显示训练中的   │  TRAINING   │
│ ▶ 开始训练  │    高斯点云]         │  INFO       │
│ ⏹ 停止训练  │                      │             │
│             │   窗口标题：          │  状态: 训练中│
│ 状态: 训练中│   [训练迭代: 7000]   │  迭代: 7000 │
│ 进度: 7000  │                      │  Loss: 0.01 │
│ Loss: 0.01  │                      │  点数: 50K  │
│ 点数: 50,000│                      │             │
└─────────────┴──────────────────────┴─────────────┘
```

## 📈 性能优化

- **输出缓冲**：使用 `bufsize=1` 和 `universal_newlines=True` 确保实时输出
- **状态节流**：限制UI更新频率为0.5秒一次
- **异步检测**：PLY文件检测在后台线程中进行
- **避免重复加载**：记录最后加载的PLY路径，避免重复加载

## 🐛 调试信息

训练过程中会在控制台输出：
```
启动训练命令: python gaussian-splatting/train.py -s ... -m ...
Training progress:  3%|▎         | 1000/30000 [00:42<19:58, 24.18it/s, Loss=0.123456]
检测到保存检查点: iteration 7000
检测到新的checkpoint: iteration 7000
加载训练可视化: 迭代 7000, 文件: output/point_cloud/iteration_7000/point_cloud.ply
```

## ✨ 下一步改进建议

1. **图表显示**：添加Loss曲线图
2. **更多指标**：PSNR、SSIM等质量指标
3. **训练日志**：保存完整的训练日志到文件
4. **暂停/恢复**：支持训练暂停和恢复
5. **多任务训练**：支持同时运行多个训练任务

## 🎉 总结

现在你可以：
- ✅ 实时看到训练过程中的Loss、点数等指标
- ✅ 在右侧面板监控训练状态
- ✅ 在播放窗口实时查看训练中的高斯点云
- ✅ 通过窗口标题和toast提示了解训练进度
- ✅ 体验更流畅的训练可视化体验！

开始享受实时训练可视化吧！🚀
