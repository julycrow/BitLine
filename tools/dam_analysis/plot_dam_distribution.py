#!/usr/bin/env python3
"""
CGNet DAM分布可视化工具

绘制类似Sparse DETR的DAM非零值比例分布柱状图。
对于每个解码器层，展示有多少样本的DAM非零值比例落在不同区间。

使用方法:
python plot_dam_distribution.py --input dam_analysis_results.json --output dam_distribution_plots
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Use default font (no Chinese font needed)
rcParams['axes.unicode_minus'] = False  # Properly display minus sign


def plot_dam_distribution_per_layer(layer_data, layer_name, output_dir, num_bins=50):
    """
    为单个解码器层绘制DAM非零值比例分布图
    
    Args:
        layer_data: 该层所有样本的引用比例列表
        layer_name: 层名称（如 'layer_0'）
        output_dir: 输出目录
        num_bins: 柱状图的bin数量
    """
    if len(layer_data) == 0:
        print(f"警告: {layer_name} 没有数据，跳过绘图")
        return
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 计算统计信息
    mean_ratio = np.mean(layer_data)
    std_ratio = np.std(layer_data)
    min_ratio = np.min(layer_data)
    max_ratio = np.max(layer_data)
    
    # 绘制柱状图
    n, bins, patches = ax.hist(layer_data, bins=num_bins, 
                                range=(0.0, 1.0),
                                color='skyblue', 
                                edgecolor='black', 
                                alpha=0.7)
    
    # 根据比例值为柱子着色（类似Sparse DETR的渐变色）
    # 使用渐变色：蓝色->青色->绿色->黄色
    cm = plt.cm.get_cmap('viridis')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    # 添加平均值的垂直线
    ax.axvline(mean_ratio, color='red', linestyle='--', linewidth=2, 
               label=f'μ={mean_ratio:.2f}')
    
    # 添加标签和标题
    ax.set_xlabel('Ratio of Non-zero Values of Decoder Attention Map', 
                  fontsize=14)
    ax.set_ylabel('# of Samples', fontsize=14)
    ax.set_title(f'{layer_name.replace("_", " ").title()} - DAM Distribution\n'
                f'μ={mean_ratio:.3f}, σ={std_ratio:.3f}, '
                f'Range=[{min_ratio:.3f}, {max_ratio:.3f}]',
                fontsize=16, pad=20)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    ax.legend(fontsize=12)
    
    # 设置x轴范围
    ax.set_xlim(0.0, 1.0)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'{layer_name}_dam_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存: {output_path}")
    
    plt.close()


def plot_all_layers_combined(all_layers_data, output_dir, num_bins=50):
    """
    绘制所有层的综合对比图
    
    Args:
        all_layers_data: 字典，键为层名，值为该层的引用比例列表
        output_dir: 输出目录
        num_bins: 柱状图的bin数量
    """
    num_layers = len(all_layers_data)
    if num_layers == 0:
        print("警告: 没有数据，跳过综合绘图")
        return
    
    # 创建子图
    rows = (num_layers + 1) // 2  # 2列布局
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    
    # 展平axes数组以便迭代
    if num_layers > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # 为每一层绘图
    for idx, (layer_name, layer_data) in enumerate(sorted(all_layers_data.items())):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # 计算统计信息
        mean_ratio = np.mean(layer_data)
        
        # 绘制柱状图
        n, bins, patches = ax.hist(layer_data, bins=num_bins, 
                                    range=(0.0, 1.0),
                                    color='skyblue', 
                                    edgecolor='black', 
                                    alpha=0.7)
        
        # 着色
        cm = plt.cm.get_cmap('viridis')
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        for c, p in zip(bin_centers, patches):
            plt.setp(p, 'facecolor', cm(c))
        
        # 添加平均值线
        ax.axvline(mean_ratio, color='red', linestyle='--', linewidth=2, 
                   label=f'μ={mean_ratio:.2f}')
        
        # 设置标题和标签
        ax.set_title(f'{layer_name.replace("_", " ").title()}', fontsize=14)
        ax.set_xlabel('DAM Ratio', fontsize=11)
        ax.set_ylabel('# of Samples', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.0, 1.0)
    
    # 隐藏多余的子图
    for idx in range(num_layers, len(axes)):
        axes[idx].set_visible(False)
    
    # 添加总标题
    fig.suptitle('DAM Distribution Comparison Across Decoder Layers', fontsize=18, y=1.00)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'all_layers_dam_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存综合图: {output_path}")
    
    plt.close()


def plot_layer_comparison_box(all_layers_data, output_dir):
    """
    绘制各层DAM比例的箱线图对比
    
    Args:
        all_layers_data: 字典，键为层名，值为该层的引用比例列表
        output_dir: 输出目录
    """
    if len(all_layers_data) == 0:
        print("警告: 没有数据，跳过箱线图")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 准备数据
    layer_names = sorted(all_layers_data.keys())
    layer_data_list = [all_layers_data[name] for name in layer_names]
    
    # 绘制箱线图
    bp = ax.boxplot(layer_data_list, labels=layer_names, 
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    
    # 为箱子着色
    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 设置标签
    ax.set_xlabel('Decoder Layer', fontsize=14)
    ax.set_ylabel('DAM Ratio', fontsize=14)
    ax.set_title('DAM Ratio Distribution Comparison Across Decoder Layers (Boxplot)', fontsize=16, pad=20)
    
    # 旋转x轴标签
    plt.xticks(rotation=45)
    
    # 添加网格
    ax.grid(True, alpha=0.3, axis='y')
    
    # 设置y轴范围
    ax.set_ylim(0.0, 1.0)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'layers_comparison_boxplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存箱线图: {output_path}")
    
    plt.close()


def load_dam_results(json_path):
    """
    从JSON文件加载DAM分析结果
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        dict: 包含各层数据的字典
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def extract_layer_ratios(dam_results):
    """
    从DAM结果中提取每一层每个样本的引用比例
    
    注意：这需要修改test_dam_analysis.py来保存每个样本的详细数据
    目前只能从统计数据中提取有限信息
    
    Args:
        dam_results: DAM分析结果字典
        
    Returns:
        dict: 每层的引用比例列表
    """
    # TODO: 这里需要从修改后的分析脚本中读取详细的样本级数据
    # 现在只能提供示例数据
    
    print("警告: 当前JSON文件可能不包含样本级别的详细数据")
    print("建议运行修改后的DAM分析脚本以保存详细数据")
    
    return {}


def parse_args():
    parser = argparse.ArgumentParser(description='绘制CGNet DAM分布图')
    parser.add_argument('--input', type=str, 
                       default='tools/dam_analysis/outputs/dam_analysis_results.json',
                       help='DAM分析结果JSON文件路径')
    parser.add_argument('--output', type=str, 
                       default='tools/dam_analysis/outputs/dam_distribution_plots',
                       help='输出图片目录')
    parser.add_argument('--bins', type=int, default=50,
                       help='柱状图的bin数量')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*80)
    print("CGNet DAM分布可视化工具")
    print("="*80)
    print(f"\n输入文件: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"Bin数量: {args.bins}\n")
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 加载DAM结果
    if not os.path.exists(args.input):
        print(f"❌ 错误: 找不到文件 {args.input}")
        print("\n请先运行DAM分析:")
        print("  python test_dam_analysis.py --max-samples 500")
        return
    
    print(f"正在加载DAM结果...")
    dam_results = load_dam_results(args.input)
    
    # 检查是否包含样本级数据
    if 'sample_level_data' in dam_results:
        # 如果有样本级数据，直接使用
        all_layers_data = dam_results['sample_level_data']
        print(f"✅ 找到样本级数据，共 {len(all_layers_data)} 层")
    else:
        print("❌ JSON文件不包含样本级数据")
        print("\n需要修改test_dam_analysis.py来保存详细的样本级数据")
        print("正在为您生成修改后的分析脚本...")
        print("\n请运行: python test_dam_analysis_detailed.py --max-samples 500")
        return
    
    # 为每一层绘制分布图
    print("\n开始绘制各层分布图...")
    for layer_name, layer_data in sorted(all_layers_data.items()):
        print(f"正在绘制 {layer_name}...")
        plot_dam_distribution_per_layer(layer_data, layer_name, args.output, args.bins)
    
    # 绘制综合对比图
    print("\n绘制综合对比图...")
    plot_all_layers_combined(all_layers_data, args.output, args.bins)
    
    # 绘制箱线图
    print("\n绘制箱线图...")
    plot_layer_comparison_box(all_layers_data, args.output)
    
    print("\n" + "="*80)
    print("✅ 所有图表已生成!")
    print(f"图表保存在: {args.output}/")
    print("="*80)


if __name__ == '__main__':
    main()
