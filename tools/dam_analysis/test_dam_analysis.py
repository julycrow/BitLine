#!/usr/bin/env python3
"""
CGNet DAM (Decoder cross-Attention Map) 分析工具

这个脚本用于分析CGNet模型中解码器对象查询引用了多少编码器输出标记。
基于Sparse DETR的DAM分析思路，帮助理解模型的稀疏性和效率。

DAM分析流程:
1. 加载训练好的CGNet模型
2. 在验证集上进行推理
3. 收集解码器cross-attention的注意力权重
4. 统计每个编码器token被引用的次数
5. 计算DAM非零值的比例（被引用的编码器token比例）

使用方法:
python test_dam_analysis.py

参考: Sparse DETR - "Efficient DETR: Improving End-to-End Object Detector with Dense Prior"
"""

import argparse
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


class DAMAnalyzer:
    """
    DAM (Decoder cross-Attention Map) 分析器
    
    用于分析解码器对象查询引用的编码器token数量和比例
    """
    
    def __init__(self, model, dataloader, device='cuda:0'):
        """
        初始化DAM分析器
        
        Args:
            model: CGNet模型
            dataloader: 数据加载器
            device: 计算设备
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        
        # DAM统计信息
        self.statistics = {
            'total_samples': 0,
            'total_encoder_tokens': 0,
            'total_referenced_tokens': 0,
            'layer_wise_stats': {},  # 每层的统计
            'sample_wise_ratios': [],  # 每个样本的引用比例
        }
    
    def enable_dam_analysis(self):
        """
        启用模型的DAM分析模式
        
        遍历模型，找到支持DAM分析的模块并启用它们
        """
        print("正在启用DAM分析模式...")
        
        # 启用解码器的DAM分析
        decoder = self._find_decoder()
        if decoder is not None and hasattr(decoder, 'enable_dam_analysis_mode'):
            decoder.enable_dam_analysis_mode()
            print(f"✓ 解码器DAM分析已启用: {type(decoder).__name__}")
        
        # 启用cross-attention模块的注意力权重保存
        cross_attns = self._find_cross_attentions()
        for idx, attn in enumerate(cross_attns):
            if hasattr(attn, 'enable_save_attn_weights'):
                attn.enable_save_attn_weights()
                print(f"✓ Cross-Attention模块 {idx+1}/{len(cross_attns)} 注意力权重保存已启用")
        
        print(f"DAM分析模式启用完成，共找到 {len(cross_attns)} 个cross-attention模块\n")
    
    def _find_decoder(self):
        """查找模型中的解码器模块"""
        for name, module in self.model.named_modules():
            if 'decoder' in name.lower() and 'MapTRDecoderWithDAM' in type(module).__name__:
                return module
        return None
    
    def _find_cross_attentions(self):
        """查找模型中的cross-attention模块"""
        cross_attns = []
        for name, module in self.model.named_modules():
            # 查找CustomMSDeformableAttentionWithDAM模块
            if 'CustomMSDeformableAttentionWithDAM' in type(module).__name__:
                cross_attns.append(module)
        return cross_attns
    
    def analyze_batch(self, data):
        """
        分析一个batch的数据
        
        Args:
            data: 输入数据batch
            
        Returns:
            dict: 该batch的DAM统计信息
        """
        # 前向传播（推理模式）
        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **data)
        
        # 从模型中收集DAM统计
        batch_stats = self._collect_dam_statistics()
        
        return batch_stats
    
    def _collect_dam_statistics(self):
        """
        从模型中收集DAM统计信息
        
        Returns:
            dict: 收集到的统计信息
        """
        stats = {}
        
        # 从cross-attention模块收集注意力权重
        cross_attns = self._find_cross_attentions()
        
        for layer_idx, attn in enumerate(cross_attns):
            if hasattr(attn, 'get_last_attn_weights'):
                attn_weights = attn.get_last_attn_weights()
                
                if attn_weights is not None:
                    # 分析注意力权重
                    layer_stat = self._analyze_attention_weights(attn_weights)
                    stats[f'layer_{layer_idx}'] = layer_stat
        
        return stats
    
    def _analyze_attention_weights(self, attn_weights, threshold=0):
        """
        分析注意力权重，计算DAM统计
        
        对于Deformable DETR，注意力不是对所有编码器token计算的，而是通过采样实现的。
        我们需要：
        1. 将采样位置映射回编码器token的索引
        2. 统计哪些编码器token被采样到（被引用）
        
        Args:
            attn_weights: 可以是字典（包含sampling_locations等）或张量
            threshold: 判断为"被引用"的阈值
            
        Returns:
            dict: 统计信息
        """
        if attn_weights is None:
            return None
        
        # 检查是否是Deformable Attention的字典格式
        if isinstance(attn_weights, dict):
            sampling_locations = attn_weights['sampling_locations']  # [bs, num_query, num_heads, num_levels, num_points, 2] torch.Size([1, 1000, 8, 1, 4, 2])
            attention_weights_val = attn_weights['attention_weights']  # [bs, num_query, num_heads, num_levels, num_points] torch.Size([1, 1000, 8, 1, 4])
            spatial_shapes = attn_weights['spatial_shapes']  # [num_levels, 2] torch.Size([1, 2])
            
            bs, num_query, num_heads, num_levels, num_points, _ = sampling_locations.shape
            
            # 计算总的编码器token数（所有level的特征图大小之和）
            total_encoder_tokens = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item()  # 200*100=20000
            
            # 对于每个样本，统计被引用的编码器token
            referenced_tokens_list = []
            
            for b in range(bs):
                # 创建一个集合来记录被采样到的编码器位置
                sampled_positions = set()
                
                for level_idx in range(num_levels):
                    h, w = spatial_shapes[level_idx]
                    level_start = (spatial_shapes[:level_idx, 0] * spatial_shapes[:level_idx, 1]).sum().item() if level_idx > 0 else 0
                    
                    # 获取这个level的采样位置 [num_query, num_heads, num_points, 2] torch.Size([1000, 8, 4, 2])
                    level_sampling_locs = sampling_locations[b, :, :, level_idx, :, :]
                    # 获取这个level的注意力权重 [num_query, num_heads, num_points]
                    level_attn_weights = attention_weights_val[b, :, :, level_idx, :]
                    
                    # 将归一化坐标[0,1]转换为特征图坐标
                    # level_sampling_locs: [num_query, num_heads, num_points, 2]
                    grid_y = (level_sampling_locs[..., 1] * h).long().clamp(0, h-1)  # [num_query, num_heads, num_points]
                    grid_x = (level_sampling_locs[..., 0] * w).long().clamp(0, w-1)  # [num_query, num_heads, num_points]
                    
                    # 计算在整个特征图中的索引
                    grid_indices = grid_y * w + grid_x  # [num_query, num_heads, num_points]
                    
                    # 加上level的起始偏移
                    global_indices = grid_indices + level_start
                    
                    # 展平并过滤：只记录注意力权重大于阈值的位置
                    # 首先展平
                    flat_indices = global_indices.reshape(-1)  # [num_query * num_heads * num_points]
                    flat_weights = level_attn_weights.reshape(-1)  # [num_query * num_heads * num_points]
                    
                    # 过滤
                    valid_mask = flat_weights > threshold
                    valid_indices = flat_indices[valid_mask]
                    
                    # 添加到集合中（自动去重）
                    sampled_positions.update(valid_indices.cpu().numpy().tolist())
                
                # 记录这个样本被引用的token数
                referenced_tokens_list.append(len(sampled_positions))
            
            # 转换为tensor以便计算统计
            import torch
            referenced_tokens = torch.tensor(referenced_tokens_list, dtype=torch.float32)
            reference_ratio = referenced_tokens / total_encoder_tokens
            
            return {
                'total_encoder_tokens': total_encoder_tokens,
                'referenced_tokens_mean': referenced_tokens.mean().item(),
                'referenced_tokens_std': referenced_tokens.std().item(),
                'reference_ratio_mean': reference_ratio.mean().item(),
                'reference_ratio_std': reference_ratio.std().item(),
                'reference_ratio_min': reference_ratio.min().item(),
                'reference_ratio_max': reference_ratio.max().item(),
            }
        
        # 如果不是deformable attention的格式，使用原来的简单方法
        # 处理不同形状的注意力权重
        if len(attn_weights.shape) == 5:
            # [bs, num_query, num_heads, num_levels, num_points]
            bs, num_query, num_heads, num_levels, num_points = attn_weights.shape
            # 重塑为 [bs, num_query, num_heads * num_levels * num_points]
            attn_weights = attn_weights.reshape(bs, num_query, -1)
        elif len(attn_weights.shape) == 4:
            # [bs, num_query, num_heads, num_keys]
            bs, num_query, num_heads, num_keys = attn_weights.shape
            # 重塑为 [bs, num_query, num_heads * num_keys]
            attn_weights = attn_weights.reshape(bs, num_query, -1)
        
        bs, num_query, num_encoder_tokens = attn_weights.shape
        
        # 对所有queries求和，得到每个encoder token的总注意力
        # [bs, num_encoder_tokens]
        encoder_token_attention = attn_weights.sum(dim=1)
        
        # 统计被引用的encoder tokens（注意力 > threshold）
        referenced_mask = encoder_token_attention > threshold
        referenced_tokens = referenced_mask.float().sum(dim=1)  # [bs]
        total_tokens = num_encoder_tokens
        
        # 计算引用比例
        reference_ratio = referenced_tokens / total_tokens
        
        return {
            'total_encoder_tokens': total_tokens,
            'referenced_tokens_mean': referenced_tokens.mean().item(),
            'referenced_tokens_std': referenced_tokens.std().item(),
            'reference_ratio_mean': reference_ratio.mean().item(),
            'reference_ratio_std': reference_ratio.std().item(),
            'reference_ratio_min': reference_ratio.min().item(),
            'reference_ratio_max': reference_ratio.max().item(),
        }
    
    def run_analysis(self, max_samples=None):
        """
        运行DAM分析
        
        Args:
            max_samples: 最大分析样本数，None表示分析全部
            
        Returns:
            dict: 完整的DAM分析结果
        """
        print(f"开始DAM分析...")
        print(f"数据集大小: {len(self.dataloader.dataset)}")
        if max_samples:
            print(f"最大分析样本数: {max_samples}\n")
        
        # 启用DAM分析模式
        self.enable_dam_analysis()
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 遍历数据集
        sample_count = 0
        layer_stats_accumulator = {}
        
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.dataloader, desc="分析中")):
                # 检查是否达到最大样本数
                if max_samples and sample_count >= max_samples:
                    break
                
                # 分析当前batch
                batch_stats = self.analyze_batch(data)
                
                # 累积统计信息
                for layer_name, layer_stat in batch_stats.items():
                    if layer_stat is None:
                        continue
                    
                    if layer_name not in layer_stats_accumulator:
                        layer_stats_accumulator[layer_name] = {
                            'reference_ratios': [],
                            'referenced_tokens': [],
                        }
                    
                    layer_stats_accumulator[layer_name]['reference_ratios'].append(
                        layer_stat['reference_ratio_mean'])
                    layer_stats_accumulator[layer_name]['referenced_tokens'].append(
                        layer_stat['referenced_tokens_mean'])
                
                # 简单地增加1，因为每个batch只有1个样本（samples_per_gpu=1）
                sample_count += 1
        
        # 计算最终统计（包括sample-level数据）
        final_stats = self._compute_final_statistics(layer_stats_accumulator)
        final_stats['total_samples_analyzed'] = sample_count
        
        return final_stats
    
    def _compute_final_statistics(self, layer_stats_accumulator):
        """
        计算最终的DAM统计信息
        
        Args:
            layer_stats_accumulator: 累积的层级统计信息
            
        Returns:
            dict: 最终统计结果（包含sample_level_data用于绘图）
        """
        final_stats = {
            'layer_wise_statistics': {},
            'overall_statistics': {},
            'sample_level_data': {},  # 新增：样本级数据用于绘制分布图
        }
        
        all_ratios = []
        
        for layer_name, layer_data in layer_stats_accumulator.items():
            ratios = layer_data['reference_ratios']
            tokens = layer_data['referenced_tokens']
            
            if len(ratios) > 0:
                layer_stats = {
                    'mean_reference_ratio': np.mean(ratios),
                    'std_reference_ratio': np.std(ratios),
                    'min_reference_ratio': np.min(ratios),
                    'max_reference_ratio': np.max(ratios),
                    'mean_referenced_tokens': np.mean(tokens),
                    'std_referenced_tokens': np.std(tokens),
                }
                final_stats['layer_wise_statistics'][layer_name] = layer_stats
                all_ratios.extend(ratios)
                
                # 保存sample-level数据用于绘图
                final_stats['sample_level_data'][layer_name] = ratios
        
        # 整体统计
        if len(all_ratios) > 0:
            final_stats['overall_statistics'] = {
                'mean_reference_ratio': np.mean(all_ratios),
                'std_reference_ratio': np.std(all_ratios),
                'min_reference_ratio': np.min(all_ratios),
                'max_reference_ratio': np.max(all_ratios),
            }
        
        return final_stats
    
    def print_results(self, stats):
        """
        打印DAM分析结果
        
        Args:
            stats: 统计结果字典
        """
        print("\n" + "="*80)
        print("DAM (Decoder cross-Attention Map) 分析结果")
        print("="*80)
        
        print(f"\n总分析样本数: {stats.get('total_samples_analyzed', 'N/A')}")
        
        # 整体统计
        if 'overall_statistics' in stats and stats['overall_statistics']:
            print("\n整体统计:")
            print("-" * 60)
            overall = stats['overall_statistics']
            print(f"  平均引用比例: {overall['mean_reference_ratio']:.2%} ± {overall['std_reference_ratio']:.2%}")
            print(f"  引用比例范围: [{overall['min_reference_ratio']:.2%}, {overall['max_reference_ratio']:.2%}]")
            print(f"  非零DAM值比例: {overall['mean_reference_ratio']:.2%}")
            print(f"\n  解释: 平均而言，解码器对象查询引用了约 {overall['mean_reference_ratio']:.1%} 的编码器输出标记")
        
        # 层级统计
        if 'layer_wise_statistics' in stats and stats['layer_wise_statistics']:
            print("\n各解码器层统计:")
            print("-" * 60)
            for layer_name, layer_stat in sorted(stats['layer_wise_statistics'].items()):
                print(f"\n  {layer_name}:")
                print(f"    引用比例: {layer_stat['mean_reference_ratio']:.2%} ± {layer_stat['std_reference_ratio']:.2%}")
                print(f"    引用范围: [{layer_stat['min_reference_ratio']:.2%}, {layer_stat['max_reference_ratio']:.2%}]")
                print(f"    平均引用token数: {layer_stat['mean_referenced_tokens']:.1f} ± {layer_stat['std_referenced_tokens']:.1f}")
        
        print("\n" + "="*80)
    
    def save_results(self, stats, output_path='dam_analysis_results.json'):
        """
        保存DAM分析结果到JSON文件
        
        Args:
            stats: 统计结果字典
            output_path: 输出文件路径
        """
        # 添加时间戳
        stats['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n分析结果已保存到: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='CGNet DAM Analysis')
    parser.add_argument('--config', type=str, 
                       default='projects/configs/cgnet/cgnet_ep110_dam.py',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str,
                       default='ckpts/cgnet_ep110.pth',
                       help='checkpoint文件路径')
    parser.add_argument('--max-samples', type=int, default=100,
                       help='最大分析样本数，用于快速测试')
    parser.add_argument('--output', type=str, 
                       default='tools/dam_analysis/outputs/dam_analysis_results.json',
                       help='结果输出文件路径')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='计算设备')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*80)
    print("CGNet DAM (Decoder cross-Attention Map) 分析工具")
    print("="*80)
    print(f"\n配置文件: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"设备: {args.device}\n")
    
    # 加载配置
    print("正在加载配置...")
    cfg = Config.fromfile(args.config)
    
    # 导入自定义模块
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            plg_lib = importlib.import_module(_module_path)
    
    # 构建数据集
    print("正在构建数据集...")
    dataset = build_dataset(cfg.data.test)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    print(f"数据集大小: {len(dataset)}")
    
    # 构建模型
    print("正在构建模型...")
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # 加载checkpoint
    print(f"正在加载checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    
    # 将模型放到GPU上
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    print("模型加载完成!\n")
    
    # 创建DAM分析器
    analyzer = DAMAnalyzer(model, dataloader, device=args.device)
    
    # 运行分析
    stats = analyzer.run_analysis(max_samples=args.max_samples)
    
    # 打印结果
    analyzer.print_results(stats)
    
    # 保存结果
    analyzer.save_results(stats, args.output)
    
    print("\nDAM分析完成!")


if __name__ == '__main__':
    main()
