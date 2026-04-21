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
            cmd = [
                self.python_exe,
                self.train_script,
                '-s', source_path,
                '-m', model_path,
                '--iterations', str(iterations),
                '--save_iterations', str(iterations),  # 只在最后保存
                '--test_iterations', str(iterations),
            ]
            
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
            
            # 解析输出
            iteration_pattern = re.compile(r'Iteration (\d+)')
            loss_pattern = re.compile(r'Loss: ([\d.]+)')
            points_pattern = re.compile(r'Points: (\d+)')
            
            check_counter = 0
            
            for line in iter(self.process.stdout.readline, ''):
                if self.should_stop:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                print(line)  # 输出到控制台
                
                # 解析迭代信息
                iter_match = iteration_pattern.search(line)
                if iter_match:
                    self.current_iteration = int(iter_match.group(1))
                
                # 解析loss
                loss_match = loss_pattern.search(line)
                if loss_match:
                    self.current_loss = float(loss_match.group(1))
                
                # 解析点数
                points_match = points_pattern.search(line)
                num_points = int(points_match.group(1)) if points_match else 0
                
                # 更新状态
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
                    if check_counter >= 10:  # 每10行输出检查一次
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
            return
        
        try:
            # 查找 point_cloud/iteration_* 目录
            point_cloud_dir = os.path.join(self.model_path, "point_cloud")
            if not os.path.exists(point_cloud_dir):
                return
            
            # 获取所有iteration目录
            iteration_dirs = glob.glob(os.path.join(point_cloud_dir, "iteration_*"))
            if not iteration_dirs:
                return
            
            # 找到最新的迭代
            iterations = []
            for d in iteration_dirs:
                try:
                    iter_num = int(os.path.basename(d).split('_')[1])
                    iterations.append((iter_num, d))
                except:
                    continue
            
            if not iterations:
                return
            
            # 按迭代次数排序，取最新的
            iterations.sort()
            latest_iter, latest_dir = iterations[-1]
            
            # 如果是新的迭代，加载PLY文件
            if latest_iter > self.last_loaded_iteration:
                ply_path = os.path.join(latest_dir, "point_cloud.ply")
                if os.path.exists(ply_path):
                    self.last_loaded_iteration = latest_iter
                    if self.on_visualization_update_callback:
                        self.on_visualization_update_callback(ply_path, latest_iter)
                    print(f"检测到新的checkpoint: iteration {latest_iter}")
        
        except Exception as e:
            print(f"检查PLY文件时出错: {e}")
    
    def get_status(self):
        """获取训练状态"""
        statuses = []
        while not self.status_queue.empty():
            statuses.append(self.status_queue.get())
        return statuses
