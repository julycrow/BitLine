#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CGNet 参数量和FLOPs并排对比工具
"""

import subprocess
import re
import sys

def run_analysis(script, config):
    """运行分析脚本并捕获输出"""
    cmd = ['python', script, config]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def extract_summary(text, metric_type):
    """提取摘要信息"""
    if metric_type == 'params':
        match = re.search(r'总参数量:\s+(\S+)\s+\((\S+)\)', text)
        if match:
            return f"{match.group(1)} ({match.group(2)})"
    elif metric_type == 'flops':
        match = re.search(r'总FLOPs:\s+(\S+)\s+\((\S+)\)', text)
        if match:
            return f"{match.group(1)} ({match.group(2)})"
    return "N/A"

def extract_components(text, metric_type):
    """提取组件级数据"""
    components = {}
    
    if metric_type == 'params':
        # 查找 "主要组件参数量分布 (与FLOPs层级一致)" 部分
        section_start = text.find('主要组件参数量分布 (与FLOPs层级一致)')
    else:  # flops
        # 查找 "主要组件FLOPs分布" 部分
        section_start = text.find('主要组件FLOPs分布')
    
    if section_start == -1:
        return components
    
    # 找到数据行的开始（跳过标题行）
    lines_start = text.find('-'*100, section_start)
    if lines_start == -1:
        return components
    
    # 找到数据行的结束
    lines_end = text.find('='*100, lines_start + 1)
    if lines_end == -1:
        return components
    
    # 提取数据行
    data_section = text[lines_start:lines_end].strip()
    lines = data_section.split('\n')[1:]  # 跳过分隔线
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('=') or line.startswith('-'):
            continue
        
        # 分割行，至少需要3部分: 名称 数值 百分比
        parts = line.split()
        if len(parts) >= 3:
            module_name = parts[0]
            value = parts[1]
            # 确保value是合法的数值格式
            if any(unit in value for unit in ['K', 'M', 'G', 'T']):
                components[module_name] = value
    
    return components

def main():
    config = "projects/configs/cgnet/cgnet_ep110.py"
    
    print("="*100)
    print("CGNet 模型复杂度并排对比".center(100))
    print("="*100)
    print()
    
    # 运行分析
    print("正在分析参数量...")
    params_output = run_analysis('tools/analyze_flops_params/calculate_params.py', config)
    
    print("正在分析FLOPs...")
    flops_output = run_analysis('tools/analyze_flops_params/calculate_flops.py', config)
    
    # 提取摘要
    params_summary = extract_summary(params_output, 'params')
    flops_summary = extract_summary(flops_output, 'flops')
    
    print()
    print("="*100)
    print("总体摘要".center(100))
    print("="*100)
    print(f"总参数量: {params_summary:>30}")
    print(f"总FLOPs:  {flops_summary:>30}")
    print("="*100)
    print()
    
    # 提取组件数据
    params_components = extract_components(params_output, 'params')
    flops_components = extract_components(flops_output, 'flops')
    
    # 合并所有模块名
    all_modules = sorted(set(list(params_components.keys()) + list(flops_components.keys())))
    
    # 打印对比表格
    print("="*100)
    print("模块级对比 (与FLOPs层级一致)".center(100))
    print("="*100)
    print(f"{'模块名称':<50} {'参数量':>20} {'FLOPs':>20}")
    print("-"*100)
    
    for module in all_modules:
        params = params_components.get(module, 'N/A')
        flops = flops_components.get(module, 'N/A')
        print(f"{module:<50} {params:>20} {flops:>20}")
    
    print("="*100)
    print()
    print("说明:")
    print("- 参数量: 模型权重的数量,影响模型大小和内存占用")
    print("- FLOPs: 计算量,影响推理速度")
    print()

if __name__ == '__main__':
    main()
