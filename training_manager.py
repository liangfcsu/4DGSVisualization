"""
Training Manager - 在后台运行3DGS训练并提供实时更新和可视化
通过调用 train.py 脚本并解析输出来实现
"""

import os
import sys
import subprocess
import threading
import time
import re
import glob
from queue import Queue
from pathlib import Path
from typing import Optional, Callable, Dict


class TrainingManager:
    """管理3DGS训练过程的类，支持实时可视化"""
    
    def __init__(self):
        self.training_thread = None
        self.process = None
        self.is_training = False
        self.should_stop = False
        self.current_iteration = 0
        self.total_iterations = 30000
        self.current_loss = 0.0
        self.current_psnr = 0.0
        self.status_queue = Queue()
        
        # 训练脚本路径
        self.train_script = os.path.join(os.path.dirname(__file__), "gaussian-splatting", "train.py")
        self.python_exe = sys.executable
        
        # 实时可视化相关
        self.model_path = None
        self.visualization_update_interval = 100  # 每100次迭代检查一次新的PLY
        self.last_loaded_iteration = 0
        
        # 回调函数
        self.on_iteration_callback: Optional[Callable] = None
        self.on_complete_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None
        self.on_visualization_update_callback: Optional[Callable] = None
    
    def start_training(self, 
                      source_path: str,
                      model_path: str,
                      iterations: int = 30000,
                      save_iterations: list = None,
                      test_iterations: list = None,
                      checkpoint_iterations: list = None,
                      enable_visualization: bool = True):
        """启动训练 - 调用 train.py 脚本"""
        
        if self.is_training:
            raise RuntimeError("训练已在进行中")
        
        if not os.path.exists(self.train_script):
            raise FileNotFoundError(f"找不到训练脚本: {self.train_script}")
        
        self.total_iterations = iterations
        self.should_stop = False
        self.model_path = model_path
        self.last_loaded_iteration = 0
        
        # 创建输出目录
        os.makedirs(model_path, exist_ok=True)
        
        # 在新线程中启动训练
        self.training_thread = threading.Thread(
            target=self._train_worker,
            args=(source_path, model_path, iterations, enable_visualization),
            daemon=True
        )
        self.training_thread.start()
        self.is_training = True
        
        # 发送初始状态
        self._send_status("训练已启动", "started", {
            'iteration': 0,
            'total': iterations,
            'loss': 0.0,
            'num_points': 0
        })
        print(f"[训练] 训练管理器已启动，迭代数: {iterations}")
    
    def stop_training(self):
        """停止训练"""
        self.should_stop = True
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
    
    def _train_worker(self, source_path: str, model_path: str, 
                     iterations: int, enable_visualization: bool):
        """训练工作线程 - 调用 train.py 并解析输出"""
        try:
            # 构建训练命令
            # 设置中间checkpoint保存点，以便实时可视化
            save_iters = [500, 1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000, iterations]
            
            cmd = [
                self.python_exe,
                self.train_script,
                '-s', source_path,
                '-m', model_path,
                '--iterations', str(iterations),
                '--save_iterations'
            ]
            # 添加每个保存迭代作为单独的参数
            cmd.extend(map(str, save_iters))
            
            # 添加测试迭代
            cmd.append('--test_iterations')
            cmd.extend(map(str, save_iters))
            
            print(f"启动训练命令: {' '.join(cmd)}")
            
            # 启动训练进程
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 解析输出 - 匹配tqdm进度条和其他输出
            # 匹配格式: "Training progress:  45%|████      | 13500/30000 [05:23<06:35, 41.71it/s, Loss=0.0234567, Depth Loss=0.0001234]"
            progress_pattern = re.compile(r'Training progress:\s+(\d+)%.*?(\d+)/(\d+)')
            loss_pattern = re.compile(r'Loss[=:]\s*([\d.]+)', re.IGNORECASE)
            depth_loss_pattern = re.compile(r'Depth Loss[=:]\s*([\d.]+)', re.IGNORECASE)
            saving_pattern = re.compile(r'\[ITER (\d+)\] Saving')
            points_pattern = re.compile(r'Number of points at beginning.*?(\d+)', re.IGNORECASE)
            
            check_counter = 0
            last_update_time = time.time()
            num_points = 0
            
            for line in iter(self.process.stdout.readline, ''):
                if self.should_stop:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                print(line)  # 输出到控制台
                
                # 解析进度条信息
                progress_match = progress_pattern.search(line)
                if progress_match:
                    percent = int(progress_match.group(1))
                    self.current_iteration = int(progress_match.group(2))
                    total = int(progress_match.group(3))
                
                # 解析loss
                loss_match = loss_pattern.search(line)
                if loss_match:
                    self.current_loss = float(loss_match.group(1))
                
                # 解析depth loss
                depth_loss_match = depth_loss_pattern.search(line)
                
                # 解析保存信息
                saving_match = saving_pattern.search(line)
                if saving_match:
                    iter_num = int(saving_match.group(1))
                    print(f"[训练] 检测到保存检查点: iteration {iter_num}")
                    if enable_visualization:
                        # 等待一下让文件写入完成
                        time.sleep(0.5)
                        self._check_for_new_ply()
                        print(f"[训练] 已触发PLY文件检查")
                
                # 解析点数
                points_match = points_pattern.search(line)
                if points_match:
                    num_points = int(points_match.group(1))
                
                # 定时更新状态（避免更新太频繁）
                current_time = time.time()
                if current_time - last_update_time >= 0.5:  # 每0.5秒更新一次
                    # 同时检查是否有新的PLY文件（不仅仅在保存时检查）
                    if enable_visualization:
                        check_counter += 1
                        if check_counter >= 4:  # 每2秒检查一次（0.5s * 4 = 2s）
                            self._check_for_new_ply()
                            check_counter = 0
                    last_update_time = current_time
                    
                    if self.current_iteration > 0:
                        status = {
                            'iteration': self.current_iteration,
                            'total': iterations,
                            'loss': self.current_loss,
                            'num_points': num_points
                        }
                        self._send_status("训练中", "running", status)
                        
                        if self.on_iteration_callback:
                            self.on_iteration_callback(status)
                
                # 定期检查新的PLY文件
                if enable_visualization:
                    check_counter += 1
                    if check_counter >= 20:  # 每20行输出检查一次
                        check_counter = 0
                        self._check_for_new_ply()
            
            # 等待进程结束
            return_code = self.process.wait()
            
            if return_code == 0:
                self._send_status("训练完成", "completed")
                if self.on_complete_callback:
                    self.on_complete_callback()
                # 最后检查一次PLY文件
                if enable_visualization:
                    self._check_for_new_ply()
            else:
                error_msg = f"训练进程异常退出，返回码: {return_code}"
                self._send_status(error_msg, "error")
                if self.on_error_callback:
                    self.on_error_callback(error_msg)
                    
        except Exception as e:
            error_msg = f"训练错误: {str(e)}"
            self._send_status(error_msg, "error")
            if self.on_error_callback:
                self.on_error_callback(str(e))
            import traceback
            traceback.print_exc()
        finally:
            self.is_training = False
            if self.process:
                self.process.stdout.close()
    
    def _send_status(self, message: str, status: str, data: Dict = None):
        """发送状态更新"""
        self.status_queue.put({
            'message': message,
            'status': status,
            'data': data or {}
        })
    
    def _check_for_new_ply(self):
        """检查是否有新的训练checkpoint PLY文件"""
        if not self.model_path:
            print(f"[检查PLY] model_path未设置")
            return
        
        try:
            # 查找 point_cloud/iteration_* 目录
            point_cloud_dir = os.path.join(self.model_path, "point_cloud")
            if not os.path.exists(point_cloud_dir):
                print(f"[检查PLY] point_cloud目录不存在: {point_cloud_dir}")
                print(f"[检查PLY] 等待训练创建checkpoint...")
                return
            
            # 获取所有iteration目录
            iteration_dirs = glob.glob(os.path.join(point_cloud_dir, "iteration_*"))
            if not iteration_dirs:
                print(f"[检查PLY] 未找到iteration目录")
                return
            
            print(f"[检查PLY] 找到 {len(iteration_dirs)} 个iteration目录")
            
            # 找到最新的迭代
            iterations = []
            for d in iteration_dirs:
                try:
                    iter_num = int(os.path.basename(d).split('_')[1])
                    iterations.append((iter_num, d))
                except:
                    continue
            
            if not iterations:
                print(f"[检查PLY] 没有有效的iteration目录")
                return
            
            # 按迭代次数排序，取最新的
            iterations.sort()
            latest_iter, latest_dir = iterations[-1]
            
            print(f"[检查PLY] 最新迭代: {latest_iter}, 上次加载: {self.last_loaded_iteration}")
            
            # 检查是否已经加载过这个迭代
            if latest_iter <= self.last_loaded_iteration:
                # print(f"[检查PLY] 迭代 {latest_iter} 已加载，跳过")  # 减少日志
                return
            
            # 查找PLY文件
            ply_path = os.path.join(latest_dir, "point_cloud.ply")
            if not os.path.exists(ply_path):
                print(f"[检查PLY] PLY文件不存在: {ply_path}")
                return
            
            # 更新最后加载的迭代
            self.last_loaded_iteration = latest_iter
            print(f"[检查PLY] ✓ 检测到新的checkpoint: iteration {latest_iter}, 文件: {ply_path}")
            
            # 调用回调函数
            if self.on_visualization_update_callback:
                print(f"[检查PLY] 调用可视化回调...")
                try:
                    self.on_visualization_update_callback(ply_path, latest_iter)
                    print(f"[检查PLY] ✓ 可视化更新回调已完成")
                except Exception as e:
                    print(f"[检查PLY] ⚠ 回调执行失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[检查PLY] ⚠ 警告: 可视化回调未设置！")
        
        except Exception as e:
            print(f"[检查PLY] 检查PLY文件时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def get_status(self):
        """获取训练状态"""
        statuses = []
        while not self.status_queue.empty():
            statuses.append(self.status_queue.get())
        return statuses
