#!/usr/bin/env python3
"""应用相机切换功能补丁"""
import re
import sys

def apply_patches():
    # 读取文件
    with open('player.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("正在应用补丁...")
    
    # 1. 在工具栏添加相机选择下拉框 (在 tb.addSeparator() 和 spacer = QWidget() 之间)
    pattern1 = r'(        tb\.addSeparator\(\)\n)(        spacer = QWidget\(\))'
    replacement1 = r'''\1        
        # 相机选择（如果有加载的相机）
        if self.camera.cameras_info and len(self.camera.cameras_info) > 0:
            tb.addWidget(self._lbl(" 相机位姿:"))
            self.camera_combo = QComboBox()
            self.camera_combo.addItem("自由视角")
            for i, cam in enumerate(self.camera.cameras_info):
                self.camera_combo.addItem(f"{i+1}. {cam['name']}")
            self.camera_combo.setMinimumWidth(120)
            self.camera_combo.setToolTip("选择相机位姿 (N下一个 / P跳转最近)")
            tb.addWidget(self.camera_combo)
            tb.addSeparator()
        else:
            self.camera_combo = None
        
\2'''
    
    if re.search(pattern1, content):
        content = re.sub(pattern1, replacement1, content)
        print("✓ 添加了相机选择下拉框")
    else:
        print("✗ 未找到工具栏位置（可能已修改）")
    
    # 2. 在 _connect 方法中添加相机选择回调
    pattern2 = r'(    def _connect\(self\):\n        self\.res_combo\.currentIndexChanged\.connect\(self\._on_res_changed\)\n        self\.bg_combo\.currentIndexChanged\.connect\(self\._on_bg_changed\)\n        self\.mode_combo\.currentIndexChanged\.connect\(self\._on_mode_changed\)\n)\n(        if self\.seq:)'
    replacement2 = r'''\1        
        if self.camera_combo:
            self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)

\2'''
    
    if re.search(pattern2, content):
        content = re.sub(pattern2, replacement2, content)
        print("✓ 添加了相机选择回调连接")
    else:
        print("✗ 未找到 _connect 方法位置")
    
    # 3. 添加快捷键（在现有快捷键后面）
    pattern3 = r"(        self\._register_shortcut\(Qt\.Key_4, lambda: self\._shortcut_resolution\(3\)\)\n)"
    replacement3 = r'''\1        self._register_shortcut(Qt.Key_N, self._next_camera)
        self._register_shortcut(Qt.Key_P, self._snap_to_nearest_camera)
        self._register_shortcut(Qt.Key_R, self._reset_camera)
'''
    
    if re.search(pattern3, content):
        content = re.sub(pattern3, replacement3, content)
        print("✓ 添加了 N/P/R 快捷键")
    else:
        print("✗ 未找到快捷键注册位置")
    
    # 4. 添加相机切换方法（在 _on_mode_changed 之后）
    pattern4 = r'(    def _on_mode_changed\(self, idx\):\n        modes = \[InteractiveCamera\.MODE_FPS, InteractiveCamera\.MODE_TRACKBALL, InteractiveCamera\.MODE_ORBIT\]\n        if 0 <= idx < len\(modes\):\n            self\.camera\.mode = modes\[idx\]\n            self\._request_render\(\)\n)'
    replacement4 = r'''\1    
    def _on_camera_changed(self, idx):
        """相机选择下拉框回调"""
        if idx == 0:
            # 自由视角 - 重置相机
            self.camera.current_camera_idx = -1
        else:
            # 切换到指定相机
            camera_idx = idx - 1
            if self.camera.set_camera(camera_idx):
                pass
        self._request_render()
    
    def _next_camera(self):
        """切换到下一个相机 (N键)"""
        if self.camera.cameras_info and len(self.camera.cameras_info) > 0:
            self.camera.next_camera()
            # 同步下拉框
            if self.camera_combo:
                self.camera_combo.blockSignals(True)
                self.camera_combo.setCurrentIndex(self.camera.current_camera_idx + 1)
                self.camera_combo.blockSignals(False)
            self._request_render()
    
    def _snap_to_nearest_camera(self):
        """跳转到最近的相机 (P键)"""
        if self.camera.cameras_info and len(self.camera.cameras_info) > 0:
            self.camera.snap_to_nearest_camera()
            # 同步下拉框
            if self.camera_combo:
                self.camera_combo.blockSignals(True)
                self.camera_combo.setCurrentIndex(self.camera.current_camera_idx + 1)
                self.camera_combo.blockSignals(False)
            self._request_render()

'''
    
    if re.search(pattern4, content):
        content = re.sub(pattern4, replacement4, content)
        print("✓ 添加了相机切换方法")
    else:
        print("✗ 未找到 _on_mode_changed 方法位置")
    
    # 5. 修改 _build_objects 添加 COLMAP 加载逻辑
    pattern5 = r'(    # 加载相机\n    data_path = getattr\(args, \'path\', None\)\n    if data_path:\n        cameras_json = os\.path\.join\(data_path, "cameras\.json"\)\n        cameras      = _core\.load_cameras_from_json\(cameras_json\)\n)'
    replacement5 = r'''    # 加载相机
    data_path = getattr(args, 'path', None)
    sparse_path = getattr(args, 'sparse', None)
    
    if sparse_path and os.path.isdir(sparse_path):
        # 从 COLMAP sparse 数据加载
        cameras = _core.load_cameras_from_colmap(sparse_path)
    elif data_path:
        cameras_json = os.path.join(data_path, "cameras.json")
        cameras      = _core.load_cameras_from_json(cameras_json)
    else:
        cameras = []
'''
    
    if re.search(pattern5, content):
        content = re.sub(pattern5, replacement5, content)
        print("✓ 添加了 COLMAP 加载逻辑")
    else:
        print("✗ 未找到相机加载位置")
    
    # 保存修改后的文件
    with open('player.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n补丁应用完成！")

if __name__ == '__main__':
    try:
        apply_patches()
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
