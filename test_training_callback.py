#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试训练回调函数"""

import sys
from training_manager import TrainingManager

def test_callback(ply_path, iteration):
    print(f"✅ 回调被调用！")
    print(f"   PLY路径: {ply_path}")
    print(f"   迭代: {iteration}")

print("测试训练管理器回调...")
tm = TrainingManager()
tm.model_path = "data/coffee_martini_train"
tm.on_visualization_update_callback = test_callback

print("\n开始检查PLY文件...")
tm._check_for_new_ply()

print("\n✅ 测试完成")
