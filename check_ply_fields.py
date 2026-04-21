#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查PLY文件字段的诊断脚本
"""

import sys
from plyfile import PlyData

def check_ply_fields(ply_path):
    """检查PLY文件的字段"""
    print(f"检查PLY文件: {ply_path}")
    print("=" * 60)
    
    try:
        plydata = PlyData.read(ply_path)
        vertex = plydata.elements[0]
        
        print(f"\n顶点数量: {len(vertex.data):,}")
        print(f"\n可用字段: {len(vertex.data.dtype.names)}")
        print("-" * 60)
        
        for i, field in enumerate(vertex.data.dtype.names, 1):
            print(f"{i:2d}. {field}")
        
        # 检查关键字段
        print("\n" + "=" * 60)
        print("关键字段检查:")
        print("-" * 60)
        
        required_fields = {
            'x, y, z': ['x', 'y', 'z'],
            'opacity': ['opacity'],
            'f_dc (SH特征)': ['f_dc_0', 'f_dc_1', 'f_dc_2'],
            'RGB': ['red', 'green', 'blue'],
            'scale': ['scale_0', 'scale_1', 'scale_2'],
            'rotation': ['rot_0', 'rot_1', 'rot_2', 'rot_3'],
        }
        
        for name, fields in required_fields.items():
            has_all = all(f in vertex.data.dtype.names for f in fields)
            status = "✓" if has_all else "✗"
            print(f"{status} {name:20s} - {fields}")
        
        print("=" * 60)
        
        # 建议
        has_xyz = all(f in vertex.data.dtype.names for f in ['x', 'y', 'z'])
        has_opacity = 'opacity' in vertex.data.dtype.names
        has_features = all(f in vertex.data.dtype.names for f in ['f_dc_0', 'f_dc_1', 'f_dc_2'])
        has_rgb = all(f in vertex.data.dtype.names for f in ['red', 'green', 'blue'])
        has_scale = all(f in vertex.data.dtype.names for f in ['scale_0', 'scale_1', 'scale_2'])
        has_rot = all(f in vertex.data.dtype.names for f in ['rot_0', 'rot_1', 'rot_2', 'rot_3'])
        
        print("\n分析:")
        if has_xyz and has_opacity and has_features and has_scale and has_rot:
            print("✅ 这是完整的Gaussian Splatting格式PLY文件")
        elif has_xyz and has_rgb:
            print("⚠️  这是原始点云文件（只有xyz和RGB）")
            print("   播放器会自动转换：")
            print("   • RGB → SH特征")
            print("   • 添加默认opacity (1.0)")
            print("   • 添加默认scale (-6.0)")
            print("   • 添加默认rotation ([1,0,0,0])")
        elif has_xyz:
            print("⚠️  这是基础点云文件（只有xyz）")
            print("   播放器会添加所有缺失的默认值")
        else:
            print("❌ 缺少必需的xyz坐标字段")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        ply_path = sys.argv[1]
    else:
        ply_path = '/home/lf/algorithm/4dgsbuild/4DGSVisualization/data/coffee_martini_train/input.ply'
    
    check_ply_fields(ply_path)
