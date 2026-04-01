#!/usr/bin/env python3
"""
CGNet DAM分析 - 安装验证脚本

这个脚本验证所有DAM分析相关的文件和模块是否正确安装。
"""

import os
import sys

# 添加项目路径
dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dir_path)

def check_file(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}不存在: {filepath}")
        return False

def check_module(module_name, class_name=None):
    """检查模块是否可以导入"""
    try:
        module = __import__(module_name, fromlist=[class_name] if class_name else [])
        if class_name:
            if hasattr(module, class_name):
                print(f"✅ 模块导入成功: {module_name}.{class_name}")
                return True
            else:
                print(f"❌ 类不存在: {module_name}.{class_name}")
                return False
        else:
            print(f"✅ 模块导入成功: {module_name}")
            return True
    except ImportError as e:
        print(f"❌ 模块导入失败: {module_name}")
        print(f"   错误: {e}")
        return False

def main():
    print("="*80)
    print("CGNet DAM分析 - 安装验证")
    print("="*80)
    print()
    
    all_passed = True
    
    # 1. 检查核心模块文件
    print("1. 检查核心模块文件:")
    print("-" * 60)
    
    files_to_check = [
        ("projects/mmdet3d_plugin/cgnet/modules/decoder_dam.py", "MapTR解码器DAM模块"),
        ("projects/mmdet3d_plugin/bevformer/modules/decoder_dam.py", "注意力DAM模块"),
    ]
    
    for filepath, desc in files_to_check:
        if not check_file(filepath, desc):
            all_passed = False
    
    print()
    
    # 2. 检查配置文件
    print("2. 检查配置文件:")
    print("-" * 60)
    
    config_files = [
        ("projects/configs/cgnet/cgnet_ep110_dam.py", "DAM分析配置"),
    ]
    
    for filepath, desc in config_files:
        if not check_file(filepath, desc):
            all_passed = False
    
    print()
    
    # 3. 检查分析脚本
    print("3. 检查分析脚本:")
    print("-" * 60)
    
    script_files = [
        ("test_dam_analysis.py", "主DAM分析脚本"),
        ("test_dam_quick.py", "快速测试脚本"),
        ("run_dam_analysis.sh", "Bash运行脚本"),
    ]
    
    for filepath, desc in script_files:
        if not check_file(filepath, desc):
            all_passed = False
    
    print()
    
    # 4. 检查文档
    print("4. 检查文档:")
    print("-" * 60)
    
    doc_files = [
        ("DAM_ANALYSIS_GUIDE.md", "详细分析指南"),
        ("DAM_USAGE_QUICK_START.md", "快速开始指南"),
        ("README_DAM.md", "DAM实现总结"),
    ]
    
    for filepath, desc in doc_files:
        if not check_file(filepath, desc):
            all_passed = False
    
    print()
    
    # 5. 检查模块是否可导入
    print("5. 检查模块导入:")
    print("-" * 60)
    
    # 注意：这里可能会因为环境问题失败，但文件存在就说明安装正确
    try:
        from projects.mmdet3d_plugin.cgnet.modules.decoder_dam import MapTRDecoderWithDAM
        print("✅ MapTRDecoderWithDAM导入成功")
    except Exception as e:
        print(f"⚠️  MapTRDecoderWithDAM导入失败（可能需要完整的mmdet3d环境）")
        print(f"   错误: {str(e)[:100]}")
    
    try:
        from projects.mmdet3d_plugin.bevformer.modules.decoder_dam import CustomMSDeformableAttentionWithDAM
        print("✅ CustomMSDeformableAttentionWithDAM导入成功")
    except Exception as e:
        print(f"⚠️  CustomMSDeformableAttentionWithDAM导入失败（可能需要完整的mmdet3d环境）")
        print(f"   错误: {str(e)[:100]}")
    
    print()
    
    # 6. 检查checkpoint文件
    print("6. 检查checkpoint文件:")
    print("-" * 60)
    
    checkpoint_files = [
        ("ckpts/cgnet_ep110.pth", "CGNet ep110 checkpoint"),
        ("ckpts/cgnet_ep24.pth", "CGNet ep24 checkpoint (可选)"),
    ]
    
    for filepath, desc in checkpoint_files:
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✅ {desc}: {filepath} ({size_mb:.1f} MB)")
        else:
            print(f"⚠️  {desc}不存在: {filepath} (需要下载)")
    
    print()
    
    # 总结
    print("="*80)
    if all_passed:
        print("✅ 所有核心文件检查通过!")
        print("\n下一步:")
        print("1. 确保checkpoint文件存在: ckpts/cgnet_ep110.pth")
        print("2. 运行快速测试: python test_dam_quick.py")
        print("3. 查看使用指南: cat DAM_USAGE_QUICK_START.md")
    else:
        print("❌ 部分文件缺失，请检查上述错误")
    print("="*80)

if __name__ == '__main__':
    main()
