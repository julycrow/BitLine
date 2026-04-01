#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算CGNet模型的参数量
包括总参数量和各个模块的参数量详细信息
"""

import argparse
import torch
import sys
import os
import warnings

# 添加项目路径
# 脚本在 tools/analyze_flops_params/ 中，需要回到项目根目录
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(dir_path)

from mmcv import Config
from mmdet3d.models import build_model
import mmcv
import mmdet
import mmdet3d


def count_parameters(model):
    """
    计算模型总参数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        total_params: 总参数量
        trainable_params: 可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def count_module_parameters(model, module_name="", max_depth=3, current_depth=0):
    """
    递归计算每个模块的参数量
    
    Args:
        model: PyTorch模型或模块
        module_name: 模块名称
        max_depth: 最大递归深度
        current_depth: 当前递归深度
        
    Returns:
        results: 包含模块参数信息的字典列表
    """
    results = []
    
    # 计算当前模块的参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    direct_params = sum(p.numel() for p in model.parameters(recurse=False))
    
    if total_params > 0:
        results.append({
            'name': module_name if module_name else 'Model',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'direct_params': direct_params,
            'depth': current_depth
        })
    
    # 如果还没达到最大深度，递归处理子模块
    if current_depth < max_depth:
        for name, child in model.named_children():
            child_name = f"{module_name}.{name}" if module_name else name
            child_results = count_module_parameters(child, child_name, max_depth, current_depth + 1)
            results.extend(child_results)
    
    return results


def format_number(num):
    """格式化数字，添加千分位分隔符"""
    return f"{num:,}"


def format_size(num):
    """格式化参数量为M(百万)或K(千)单位"""
    if num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def print_parameters_table(results, show_direct=True):
    """
    打印参数量表格
    
    Args:
        results: 模块参数信息列表
        show_direct: 是否显示直接参数量
    """
    print("\n" + "="*120)
    print("模块参数量统计".center(120))
    print("="*120)
    
    # 表头
    if show_direct:
        header = f"{'模块名称':<60} {'总参数量':>15} {'可训练参数':>15} {'直接参数':>15} {'占比':>10}"
    else:
        header = f"{'模块名称':<60} {'总参数量':>20} {'可训练参数':>20} {'占比':>15}"
    print(header)
    print("-"*120)
    
    # 获取总参数量用于计算占比
    total = results[0]['total_params'] if results else 0
    
    # 按深度和参数量排序
    sorted_results = sorted(results, key=lambda x: (x['depth'], -x['total_params']))
    
    for item in sorted_results:
        indent = "  " * item['depth']
        name = indent + item['name'].split('.')[-1]
        total_params = item['total_params']
        trainable_params = item['trainable_params']
        direct_params = item['direct_params']
        percentage = (total_params / total * 100) if total > 0 else 0
        
        if show_direct:
            row = f"{name:<60} {format_size(total_params):>15} {format_size(trainable_params):>15} {format_size(direct_params):>15} {percentage:>9.2f}%"
        else:
            row = f"{name:<60} {format_size(total_params):>20} {format_size(trainable_params):>20} {percentage:>14.2f}%"
        print(row)
    
    print("="*120)


def print_summary(total_params, trainable_params):
    """打印参数量摘要"""
    print("\n" + "="*120)
    print("参数量摘要".center(120))
    print("="*120)
    print(f"总参数量:        {format_number(total_params):>20} ({format_size(total_params)})")
    print(f"可训练参数量:    {format_number(trainable_params):>20} ({format_size(trainable_params)})")
    print(f"不可训练参数量:  {format_number(total_params - trainable_params):>20} ({format_size(total_params - trainable_params)})")
    print("="*120)


def print_main_modules(results):
    """打印主要模块的参数量"""
    print("\n" + "="*120)
    print("主要模块参数量".center(120))
    print("="*120)
    
    # 获取深度为1的模块(主要模块)
    main_modules = [r for r in results if r['depth'] == 1]
    main_modules = sorted(main_modules, key=lambda x: -x['total_params'])
    
    total = results[0]['total_params'] if results else 0
    
    print(f"{'模块名称':<40} {'参数量':>20} {'占比':>15}")
    print("-"*120)
    
    for module in main_modules:
        name = module['name']
        params = module['total_params']
        percentage = (params / total * 100) if total > 0 else 0
        print(f"{name:<40} {format_size(params):>20} {percentage:>14.2f}%")
    
    print("="*120)


def aggregate_module_params(module_results):
    """聚合模块参数到主要组件（与FLOPs统计层级一致）"""
    aggregated = {}
    processed = set()  # 记录已处理的参数，避免重复计数
    
    for result in module_results:
        name = result['name']
        params = result['total_params']
        
        # 跳过根节点
        if name == 'Model':
            continue
        
        parts = name.split('.')
        
        # 确定主模块名称和是否应该计数
        should_count = False
        main_module = None
        
        # 一级模块：直接统计（但跳过pts_bbox_head，因为它会被分解为子模块）
        if len(parts) == 1:
            if parts[0] != 'pts_bbox_head':
                main_module = parts[0]
                should_count = True
            # pts_bbox_head跳过，只统计它的子模块
        # pts_bbox_head的子模块：特殊处理，保留完整层级
        elif parts[0] == 'pts_bbox_head':
            if len(parts) == 2:
                # pts_bbox_head的直接子模块，如 pts_bbox_head.transformer
                if parts[1] == 'transformer':
                    # transformer需要看更深层
                    continue
                else:
                    # 其他子模块，保留完整路径
                    main_module = f"pts_bbox_head.{parts[1]}"
                    should_count = True
            elif len(parts) == 3 and parts[1] == 'transformer':
                # transformer.encoder / transformer.decoder 等，保留完整路径
                main_module = f"pts_bbox_head.transformer.{parts[2]}"
                should_count = True
            elif len(parts) > 3 and parts[1] == 'transformer':
                # transformer的更深层子模块，跳过
                continue
        
        if should_count and main_module:
            if main_module not in aggregated:
                aggregated[main_module] = 0
            aggregated[main_module] += params
    
    return aggregated


def print_aggregated_params(aggregated_params, total_params):
    """打印聚合的参数量（与FLOPs层级一致）"""
    print("\n" + "="*120)
    print("主要组件参数量分布 (与FLOPs层级一致)".center(120))
    print("="*120)
    print(f"{'组件名称':<60} {'参数量':>30} {'占比':>25}")
    print("-"*120)
    
    sorted_items = sorted(aggregated_params.items(), key=lambda x: x[1], reverse=True)
    
    for name, params in sorted_items:
        percentage = (params / total_params * 100) if total_params > 0 else 0
        print(f"{name:<60} {format_size(params):>30} {percentage:>24.2f}%")
    
    print("="*120)


def parse_args():
    parser = argparse.ArgumentParser(description='计算CGNet模型参数量')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--depth', type=int, default=3, help='模块统计的最大深度 (default: 3)')
    parser.add_argument('--show-direct', action='store_true', help='显示直接参数量')
    parser.add_argument('--detailed', action='store_true', help='显示详细的所有模块信息')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # 禁用警告
    warnings.filterwarnings('ignore')
    
    # 加载配置
    print(f"\n正在加载配置文件: {args.config}")
    cfg = Config.fromfile(args.config)
    
    # 导入插件
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]

            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(f'导入插件: {_module_path}')
            plg_lib = importlib.import_module(_module_path)
        else:
            # import dir is the dirpath for the config file
            _module_dir = os.path.dirname(args.config)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(f'导入插件: {_module_path}')
            plg_lib = importlib.import_module(_module_path)
    
    # 构建模型
    print("正在构建模型...")
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.eval()
    
    print("模型构建完成!")
    
    # 计算总参数量
    total_params, trainable_params = count_parameters(model)
    
    # 计算各模块参数量
    print(f"\n正在统计模块参数量 (深度: {args.depth})...")
    module_results = count_module_parameters(model, max_depth=args.depth)
    
    # 打印摘要
    print_summary(total_params, trainable_params)
    
    # 打印主要模块
    print_main_modules(module_results)
    
    # 聚合并打印与FLOPs层级一致的参数量分布
    aggregated = aggregate_module_params(module_results)
    print_aggregated_params(aggregated, total_params)
    
    # 打印详细信息(如果需要)
    if args.detailed:
        print_parameters_table(module_results, show_direct=args.show_direct)
    
    print("\n参数量统计完成!")


if __name__ == '__main__':
    main()
