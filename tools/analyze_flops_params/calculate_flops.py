#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算CGNet模型的FLOPs
包括总FLOPs和各个模块的FLOPs详细信息
"""

import argparse
import torch
import sys
import os
import warnings
import numpy as np

# 添加项目路径
# 脚本在 tools/analyze_flops_params/ 中，需要回到项目根目录
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(dir_path)

from mmcv import Config
from mmdet3d.models import build_model
import mmcv
import mmdet
import mmdet3d

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("警告: fvcore未安装，将使用备选方案")

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False


def format_number(num):
    """格式化数字，添加千分位分隔符"""
    return f"{num:,}"


def format_flops(flops):
    """格式化FLOPs为G/M/K单位"""
    if flops >= 1e12:
        return f"{flops/1e12:.2f}T"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f}G"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f}M"
    elif flops >= 1e3:
        return f"{flops/1e3:.2f}K"
    else:
        return str(int(flops))


def prepare_inputs(cfg, device='cuda'):
    """
    准备模型输入数据
    
    Args:
        cfg: 配置文件
        device: 设备类型
        
    Returns:
        输入数据字典
    """
    # 获取配置参数
    bev_h = cfg.model.pts_bbox_head.bev_h
    bev_w = cfg.model.pts_bbox_head.bev_w
    embed_dims = cfg.model.pts_bbox_head.transformer.embed_dims
    num_cams = 6  # nuScenes默认6个相机
    
    # 图像特征 (来自backbone+neck的输出)
    # 假设输入图像尺寸为 900x1600，经过FPN后的特征
    bs = 1
    mlvl_feats = [
        torch.randn(bs, num_cams, embed_dims, 30, 50).to(device),  # 下采样32倍
    ]
    
    # LiDAR特征 (如果使用)
    lidar_feat = None
    
    # 图像元信息
    img_metas = [{
        'scene_token': 'test_scene',
        'sample_idx': 0,
        'ego2global_translation': [0, 0, 0],
        'ego2global_rotation': [1, 0, 0, 0],
        'can_bus': torch.zeros(18).to(device),
    }]
    
    # 前一帧BEV特征 (可选)
    prev_bev = None
    
    return mlvl_feats, lidar_feat, img_metas, prev_bev


def count_flops_fvcore(model, inputs):
    """使用fvcore计算FLOPs"""
    mlvl_feats, lidar_feat, img_metas, prev_bev = inputs
    
    try:
        # FVCore需要模型接受tuple/list作为输入
        # 创建一个wrapper来处理
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, mlvl_feats):
                return self.model(mlvl_feats, None, [], None)
        
        wrapped_model = ModelWrapper(model)
        flops = FlopCountAnalysis(wrapped_model, (mlvl_feats,))
        total_flops = flops.total()
        
        # 获取模块级别的FLOPs
        module_flops = {}
        by_module = flops.by_module()
        for name, flop_count in by_module.items():
            if flop_count > 0:
                # 去除wrapper前缀
                clean_name = name.replace('model.', '', 1)
                module_flops[clean_name] = flop_count
        
        return total_flops, module_flops, flop_count_table(flops)
    except Exception as e:
        print(f"FVCore分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def count_flops_custom(model, inputs, cfg):
    """自定义FLOPs计算方法 - 针对CGNet优化"""
    mlvl_feats, lidar_feat, img_metas, prev_bev = inputs
    
    print("\n使用自定义方法计算FLOPs...")
    
    bs = mlvl_feats[0].shape[0]
    num_cams = mlvl_feats[0].shape[1]
    C, H, W = mlvl_feats[0].shape[2:]
    
    # 从配置获取参数
    bev_h = cfg.model.pts_bbox_head.bev_h
    bev_w = cfg.model.pts_bbox_head.bev_w
    embed_dims = cfg.model.pts_bbox_head.transformer.embed_dims
    num_vec = cfg.model.pts_bbox_head.num_vec
    num_pts_per_vec = cfg.model.pts_bbox_head.num_pts_per_vec
    
    # 检查是否为稀疏化模型
    rho = getattr(cfg.model.pts_bbox_head, 'rho', None)
    is_sparse = rho is not None and rho < 1.0
    if is_sparse:
        print(f"检测到稀疏化模型: rho = {rho} (保留 {rho*100:.1f}% 的BEV tokens)")
    
    module_flops = {}
    total_flops = 0
    
    # 1. Backbone (ResNet50) FLOPs估算
    # 输入: (bs * num_cams, 3, 900, 1600)
    img_h, img_w = 900, 1600  # 典型的nuScenes图像尺寸
    
    # ResNet50的FLOPs (基于标准ImageNet配置)
    # Conv1: 7x7, stride=2
    conv1_flops = bs * num_cams * (7 * 7 * 3 * 64) * (img_h // 2) * (img_w // 2)
    
    # Layer1: 3 blocks, stride=1, output: 56x56x256
    layer1_flops = bs * num_cams * 56 * 50 * (64*256 + 256*64*9 + 256*256) * 3
    
    # Layer2: 4 blocks, stride=2, output: 28x25x512
    layer2_flops = bs * num_cams * 28 * 25 * (256*512 + 512*128*9 + 512*512) * 4
    
    # Layer3: 6 blocks, stride=2, output: 14x13x1024
    layer3_flops = bs * num_cams * 14 * 13 * (512*1024 + 1024*256*9 + 1024*1024) * 6
    
    # Layer4: 3 blocks, stride=2, output: 7x7x2048
    layer4_flops = bs * num_cams * 7 * 7 * (1024*2048 + 2048*512*9 + 2048*2048) * 3
    
    backbone_flops = conv1_flops + layer1_flops + layer2_flops + layer3_flops + layer4_flops
    module_flops['img_backbone'] = backbone_flops
    total_flops += backbone_flops
    
    # 2. FPN Neck FLOPs
    # FPN的lateral和top-down操作
    fpn_flops = bs * num_cams * H * W * (2048 * embed_dims + embed_dims * embed_dims * 9) * 2
    module_flops['img_neck'] = fpn_flops
    total_flops += fpn_flops
    
    # 3. BEV Encoder (Transformer Encoder) FLOPs
    num_encoder_layers = cfg.model.pts_bbox_head.transformer.encoder.num_layers
    num_levels = 1
    num_points = 4  # 每个query的参考点数
    num_heads = 8
    
    # BEV queries: bev_h * bev_w
    num_bev_queries_full = bev_h * bev_w
    
    # 稀疏化处理: 计算有效的BEV query数量
    if is_sparse:
        # Encoder输出: 完整的BEV queries，但后续会通过mask筛选
        # Encoder计算: 仍然对所有BEV queries计算（稀疏化发生在encoder之后）
        num_bev_queries = num_bev_queries_full
        # DAM (Denoising Attention Mask) 预测器的FLOPs
        mask_predictor_dim = getattr(cfg.model.pts_bbox_head.transformer, 'mask_predictor_dim', 256)
        dam_flops = bs * num_bev_queries_full * (embed_dims * mask_predictor_dim + mask_predictor_dim * 1)
        module_flops['pts_bbox_head.transformer.mask_predictor'] = dam_flops
        total_flops += dam_flops
    else:
        num_bev_queries = num_bev_queries_full
    
    # Multi-scale deformable attention
    # Q,K,V投影
    qkv_flops = bs * num_bev_queries * embed_dims * embed_dims * 3
    
    # Attention计算
    attention_flops = bs * num_heads * num_bev_queries * num_levels * num_points * embed_dims
    
    # FFN
    ffn_dim = embed_dims * 2
    ffn_flops = bs * num_bev_queries * (embed_dims * ffn_dim + ffn_dim * embed_dims)
    
    encoder_flops = num_encoder_layers * (qkv_flops + attention_flops + ffn_flops)
    module_flops['pts_bbox_head.transformer.encoder'] = encoder_flops
    total_flops += encoder_flops
    
    # 4. Transformer Decoder FLOPs
    num_decoder_layers = cfg.model.pts_bbox_head.transformer.decoder.num_layers
    num_queries = num_vec * num_pts_per_vec  # 1000
    
    # 稀疏化后的BEV queries数量（用于decoder的cross-attention）
    if is_sparse:
        num_bev_queries_sparse = int(num_bev_queries_full * rho)
        print(f"  稀疏化后BEV queries: {num_bev_queries_full} -> {num_bev_queries_sparse} (减少 {(1-rho)*100:.1f}%)")
    else:
        num_bev_queries_sparse = num_bev_queries_full
    
    # Self-attention
    self_attn_flops = bs * num_decoder_layers * num_queries * embed_dims * embed_dims * 4
    
    # Cross-attention (query到BEV特征) - 使用稀疏化后的BEV queries数量
    cross_attn_flops = bs * num_decoder_layers * num_queries * num_bev_queries_sparse * embed_dims
    
    # FFN
    decoder_ffn_flops = bs * num_decoder_layers * num_queries * (embed_dims * ffn_dim + ffn_dim * embed_dims)
    
    decoder_flops = self_attn_flops + cross_attn_flops + decoder_ffn_flops
    module_flops['pts_bbox_head.transformer.decoder'] = decoder_flops
    total_flops += decoder_flops
    
    # 5. Detection Head FLOPs
    # 分类分支和回归分支
    cls_flops = bs * num_vec * embed_dims * embed_dims * 3  # 3层MLP
    reg_flops = bs * num_queries * embed_dims * embed_dims * 3
    module_flops['pts_bbox_head.cls_branches'] = cls_flops
    module_flops['pts_bbox_head.reg_branches'] = reg_flops
    total_flops += (cls_flops + reg_flops)
    
    # 6. GNN模块 FLOPs
    gnn_flops = bs * num_vec * num_vec * embed_dims * embed_dims
    module_flops['pts_bbox_head.vertex_inteact'] = gnn_flops
    total_flops += gnn_flops
    
    # 7. GRU模块 FLOPs
    gru_flops = bs * num_vec * embed_dims * embed_dims * 6  # 3个门 * 2
    module_flops['pts_bbox_head.GRU'] = gru_flops
    total_flops += gru_flops
    
    # 8. Topology Head FLOPs
    topo_flops = bs * num_vec * num_vec * embed_dims * embed_dims // 4
    module_flops['pts_bbox_head.lclc_branch'] = topo_flops
    total_flops += topo_flops
    
    # 9. Beizer Transform FLOPs
    beizer_flops = bs * num_vec * num_vec * embed_dims * 2
    module_flops['pts_bbox_head.beizer_transform'] = beizer_flops
    total_flops += beizer_flops
    
    return total_flops, module_flops


def count_flops_thop(model, inputs):
    """使用thop计算FLOPs"""
    mlvl_feats, lidar_feat, img_metas, prev_bev = inputs
    
    try:
        flops, params = profile(model, inputs=(mlvl_feats, lidar_feat, img_metas, prev_bev), 
                               verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.3f")
        return flops, params, flops_str, params_str
    except Exception as e:
        print(f"THOP分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def analyze_module_flops(module_flops, total_flops):
    """分析模块级FLOPs"""
    results = []
    
    # 按FLOPs排序
    sorted_modules = sorted(module_flops.items(), key=lambda x: x[1], reverse=True)
    
    for name, flops in sorted_modules:
        percentage = (flops / total_flops * 100) if total_flops > 0 else 0
        results.append({
            'name': name,
            'flops': flops,
            'percentage': percentage
        })
    
    return results


def print_flops_summary(total_flops, total_params=None):
    """打印FLOPs摘要"""
    print("\n" + "="*120)
    print("FLOPs 分析摘要".center(120))
    print("="*120)
    print(f"总FLOPs:         {format_number(int(total_flops)):>25} ({format_flops(total_flops)})")
    if total_params is not None:
        print(f"总参数量:        {format_number(int(total_params)):>25} ({format_flops(total_params)})")
    print("="*120)


def print_module_flops_table(module_results, top_n=30):
    """打印模块FLOPs表格"""
    if not module_results:
        return
    
    print("\n" + "="*120)
    print(f"主要模块FLOPs统计 (Top {min(top_n, len(module_results))})".center(120))
    print("="*120)
    print(f"{'模块名称':<80} {'FLOPs':>20} {'占比':>15}")
    print("-"*120)
    
    for i, result in enumerate(module_results[:top_n]):
        name = result['name']
        flops = result['flops']
        percentage = result['percentage']
        
        # 截断过长的名称
        if len(name) > 75:
            name = "..." + name[-72:]
        
        print(f"{name:<80} {format_flops(flops):>20} {percentage:>14.2f}%")
    
    print("="*120)


def aggregate_module_flops(module_flops):
    """聚合模块FLOPs到主要组件（保持细粒度，不合并transformer子模块）"""
    # 直接返回module_flops，不进行聚合
    # 这样可以保持transformer.encoder和transformer.decoder的分离
    return module_flops


def print_aggregated_flops(aggregated_flops, total_flops):
    """打印聚合的FLOPs"""
    print("\n" + "="*120)
    print("主要组件FLOPs分布".center(120))
    print("="*120)
    print(f"{'组件名称':<60} {'FLOPs':>30} {'占比':>25}")
    print("-"*120)
    
    sorted_items = sorted(aggregated_flops.items(), key=lambda x: x[1], reverse=True)
    
    for name, flops in sorted_items:
        percentage = (flops / total_flops * 100) if total_flops > 0 else 0
        print(f"{name:<60} {format_flops(flops):>30} {percentage:>24.2f}%")
    
    print("="*120)


def parse_args():
    parser = argparse.ArgumentParser(description='计算CGNet模型FLOPs')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='运行设备 (default: cuda)')
    parser.add_argument('--detailed', action='store_true',
                       help='显示详细的模块FLOPs信息')
    parser.add_argument('--top-n', type=int, default=30,
                       help='显示Top N的模块 (default: 30)')
    parser.add_argument('--method', default='auto', 
                       choices=['auto', 'custom', 'fvcore', 'thop'],
                       help='FLOPs计算方法 (default: auto, 推荐custom)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # 禁用警告
    warnings.filterwarnings('ignore')
    
    # 检查CUDA可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，切换到CPU")
        args.device = 'cpu'
    
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
    model = model.to(args.device)
    
    print(f"模型构建完成! 设备: {args.device}")
    
    # 准备输入数据
    print("\n正在准备输入数据...")
    inputs = prepare_inputs(cfg, args.device)
    
    # 选择FLOPs计算方法
    method = args.method
    if method == 'auto':
        # 优先使用custom方法,更准确
        method = 'custom'
    
    print(f"使用方法: {method}")
    
    # 计算FLOPs
    print("\n正在计算FLOPs...")
    
    total_flops = None
    module_flops = None
    total_params = None
    
    if method == 'custom':
        total_flops, module_flops = count_flops_custom(model, inputs, cfg)
    
    elif method == 'fvcore' and FVCORE_AVAILABLE:
        total_flops, module_flops, flops_table = count_flops_fvcore(model, inputs)
        if total_flops is not None:
            print("\nFVCore详细报告:")
            print(flops_table)
    
    elif method == 'thop' and THOP_AVAILABLE:
        total_flops, total_params, flops_str, params_str = count_flops_thop(model, inputs)
        if total_flops is not None:
            print(f"\nTHOP结果: FLOPs={flops_str}, Params={params_str}")
    
    # 打印结果
    if total_flops is not None:
        print_flops_summary(total_flops, total_params)
        
        if module_flops:
            # 聚合并打印主要组件（总是显示）
            aggregated = aggregate_module_flops(module_flops)
            print_aggregated_flops(aggregated, total_flops)
            
            if args.detailed:
                # 分析模块FLOPs（详细模式）
                module_results = analyze_module_flops(module_flops, total_flops)
                print_module_flops_table(module_results, args.top_n)
    else:
        print("\n错误: 无法计算FLOPs")
        print("请安装必要的依赖:")
        print("  pip install fvcore")
        print("  或")
        print("  pip install thop")
    
    print("\nFLOPs分析完成!")


if __name__ == '__main__':
    main()
