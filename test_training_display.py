#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练功能测试脚本
"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

from PyQt5.QtWidgets import QApplication
from ui.right_panel import RightInfoPanel
from ui.state import UIState

def test_training_info_display():
    """测试训练信息显示"""
    print("=" * 60)
    print("测试训练信息显示")
    print("=" * 60)
    
    app = QApplication(sys.argv)
    state = UIState()
    panel = RightInfoPanel(state)
    
    print("\n1. 初始状态")
    print(f"   训练卡片存在: {hasattr(panel, 'training_card')}")
    print(f"   初始状态值: {panel._v_training_status.text()}")
    
    print("\n2. 模拟训练开始")
    panel.update_training_info(
        status='训练中',
        iteration=1000,
        total=30000,
        loss=0.0123456,
        num_points=50000
    )
    
    print(f"   状态: {panel._v_training_status.text()}")
    print(f"   迭代: {panel._v_training_iteration.text()}")
    print(f"   Loss: {panel._v_training_loss.text()}")
    print(f"   点数: {panel._v_training_points.text()}")
    
    print("\n3. 模拟训练更新")
    panel.update_training_info(
        status='训练中',
        iteration=5000,
        total=30000,
        loss=0.0098765,
        num_points=75000
    )
    
    print(f"   状态: {panel._v_training_status.text()}")
    print(f"   迭代: {panel._v_training_iteration.text()}")
    print(f"   Loss: {panel._v_training_loss.text()}")
    print(f"   点数: {panel._v_training_points.text()}")
    
    print("\n4. 模拟训练结束")
    panel.update_training_info(status=None)
    
    print(f"   状态: {panel._v_training_status.text()}")
    print(f"   迭代: {panel._v_training_iteration.text()}")
    print(f"   Loss: {panel._v_training_loss.text()}")
    print(f"   点数: {panel._v_training_points.text()}")
    
    print("\n✅ 训练信息显示测试完成！")
    print("=" * 60)

if __name__ == '__main__':
    test_training_info_display()
