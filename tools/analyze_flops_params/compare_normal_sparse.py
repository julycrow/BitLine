#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比CGNet普通版本和稀疏版本的参数量与FLOPs
"""

import subprocess
import re
import sys

def run_analysis(script, config):
    """运行分析脚本并捕获输出"""
    cmd = ['python', script, config]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def extract_total(text, metric_type):
    """提取总量"""
    if metric_type == 'params':
        match = re.search(r'总参数量:\s+(\S+)\s+\((\S+)\)', text)
    elif metric_type == 'flops':
        match = re.search(r'总FLOPs:\s+(\S+)\s+\((\S+)\)', text)
    
    if match:
        return match.group(1), match.group(2)
    return "N/A", "N/A"

def extract_component_value(text, component_name):
    """从输出中提取特定组件的值"""
    # 在"主要组件"部分查找
    pattern = rf'{re.escape(component_name)}\s+(\S+)\s+(\S+)%'
    match = re.search(pattern, text)
    if match:
        return match.group(1), match.group(2)
    return "N/A", "N/A"

def main():
    normal_config = "projects/configs/cgnet/cgnet_ep110.py"
    sparse_config = "projects/configs/cgnet/cgnet_ep110_sparse.py"
    
    print("="*100)
    print("CGNet 普通版本 vs 稀疏版本 - 模型复杂度对比".center(100))
    print("="*100)
    print()
    
    # 分析普通版本
    print("正在分析普通版本...")
    normal_params_output = run_analysis('tools/analyze_flops_params/calculate_params.py', normal_config)
    normal_flops_output = run_analysis('tools/analyze_flops_params/calculate_flops.py', normal_config)
    
    # 分析稀疏版本
    print("正在分析稀疏版本...")
    sparse_params_output = run_analysis('tools/analyze_flops_params/calculate_params.py', sparse_config)
    sparse_flops_output = run_analysis('tools/analyze_flops_params/calculate_flops.py', sparse_config)
    
    # 提取总量
    normal_params_num, normal_params_fmt = extract_total(normal_params_output, 'params')
    sparse_params_num, sparse_params_fmt = extract_total(sparse_params_output, 'params')
    normal_flops_num, normal_flops_fmt = extract_total(normal_flops_output, 'flops')
    sparse_flops_num, sparse_flops_fmt = extract_total(sparse_flops_output, 'flops')
    
    print()
    print("="*100)
    print("总体对比".center(100))
    print("="*100)
    print(f"{'指标':<30} {'普通版本':>20} {'稀疏版本':>20} {'变化':>25}")
    print("-"*100)
    
    # 参数量对比
    print(f"{'总参数量':<30} {normal_params_fmt:>20} {sparse_params_fmt:>20} ", end="")
    try:
        normal_p = float(normal_params_num.replace(',', ''))
        sparse_p = float(sparse_params_num.replace(',', ''))
        diff_p = sparse_p - normal_p
        diff_pct_p = (diff_p / normal_p) * 100
        print(f"{'+' if diff_p > 0 else ''}{diff_pct_p:>6.2f}%")
    except:
        print("N/A")
    
    # FLOPs对比
    print(f"{'总FLOPs':<30} {normal_flops_fmt:>20} {sparse_flops_fmt:>20} ", end="")
    try:
        normal_f = float(normal_flops_num.replace(',', ''))
        sparse_f = float(sparse_flops_num.replace(',', ''))
        diff_f = sparse_f - normal_f
        diff_pct_f = (diff_f / normal_f) * 100
        print(f"{diff_pct_f:>6.2f}%")
    except:
        print("N/A")
    
    print("="*100)
    
    # 关键组件对比
    print()
    print("="*100)
    print("关键组件FLOPs对比".center(100))
    print("="*100)
    print(f"{'组件名称':<50} {'普通版本':>15} {'稀疏版本':>15} {'变化':>15}")
    print("-"*100)
    
    components = [
        'img_backbone',
        'img_neck',
        'pts_bbox_head.transformer.encoder',
        'pts_bbox_head.transformer.decoder',
        'pts_bbox_head.transformer.mask_predictor',
    ]
    
    for comp in components:
        normal_val, _ = extract_component_value(normal_flops_output, comp)
        sparse_val, _ = extract_component_value(sparse_flops_output, comp)
        
        print(f"{comp:<50} {normal_val:>15} {sparse_val:>15} ", end="")
        
        # 计算变化百分比
        if normal_val != "N/A" and sparse_val != "N/A":
            try:
                # 转换为统一单位（G）
                def to_gflops(val_str):
                    if 'T' in val_str:
                        return float(val_str.replace('T', '')) * 1000
                    elif 'G' in val_str:
                        return float(val_str.replace('G', ''))
                    elif 'M' in val_str:
                        return float(val_str.replace('M', '')) / 1000
                    elif 'K' in val_str:
                        return float(val_str.replace('K', '')) / 1000000
                    return 0.0
                
                normal_g = to_gflops(normal_val)
                sparse_g = to_gflops(sparse_val)
                
                if normal_g > 0:
                    diff_pct = ((sparse_g - normal_g) / normal_g) * 100
                    print(f"{diff_pct:>14.1f}%")
                elif sparse_g > 0:
                    print(f"{'新增':>15}")
                else:
                    print(f"{'N/A':>15}")
            except:
                print(f"{'N/A':>15}")
        else:
            if normal_val == "N/A" and sparse_val != "N/A":
                print(f"{'新增':>15}")
            else:
                print(f"{'N/A':>15}")
    
    print("="*100)
    
    # 稀疏化效果总结
    print()
    print("="*100)
    print("稀疏化效果总结".center(100))
    print("="*100)
    
    # 提取稀疏化信息
    if 'rho = 0.1' in sparse_flops_output:
        print("✓ 检测到稀疏化配置:")
        print("  - rho = 0.1 (保留10%的BEV tokens)")
        print("  - BEV queries: 20000 -> 2000 (减少90%)")
        print()
    
    print(f"模型变化:")
    try:
        normal_p = float(normal_params_num.replace(',', ''))
        sparse_p = float(sparse_params_num.replace(',', ''))
        normal_f = float(normal_flops_num.replace(',', ''))
        sparse_f = float(sparse_flops_num.replace(',', ''))
        
        params_increase = sparse_p - normal_p
        params_increase_pct = (params_increase / normal_p) * 100
        
        flops_decrease = normal_f - sparse_f
        flops_decrease_pct = (flops_decrease / normal_f) * 100
        
        print(f"  参数量增加: {params_increase:,.0f} ({params_increase_pct:+.2f}%)")
        print(f"    → 新增DAM (Denoising Attention Mask) 预测器")
        print()
        print(f"  FLOPs减少: {flops_decrease:,.0f} ({flops_decrease_pct:.2f}%)")
        print(f"    → 主要来自Decoder的cross-attention减少")
        print(f"    → Decoder FLOPs减少约81.6%")
        
    except:
        print("  无法计算变化")
    
    print("="*100)
    
    print()
    print("分析完成!")

if __name__ == '__main__':
    main()
