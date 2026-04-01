"""
合并两个可视化目录的图片（上下拼接）- 多进程加速版本

将两个vis_pred目录下的同名图片进行上下拼接，保存到新目录中。
使用多进程并行处理，大幅提升速度。
"""

import os
from pathlib import Path
from tqdm import tqdm
import argparse
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool, cpu_count


def merge_images_vertical(img1_path, img2_path):
    """
    垂直拼接两张图片（上下排列）
    
    Args:
        img1_path: 上方图片路径
        img2_path: 下方图片路径
        
    Returns:
        merged_img: 拼接后的图片
    """
    try:
        # 读取图片
        img1 = Image.open(str(img1_path))
        img2 = Image.open(str(img2_path))
        
        # 获取图片尺寸
        w1, h1 = img1.size
        w2, h2 = img2.size
        
        # 统一宽度（使用较大的宽度）
        max_width = max(w1, w2)
        
        # 如果宽度不同，需要调整
        if w1 != max_width:
            img1 = img1.resize((max_width, int(h1 * max_width / w1)), Image.LANCZOS)
            h1 = img1.size[1]
        if w2 != max_width:
            img2 = img2.resize((max_width, int(h2 * max_width / w2)), Image.LANCZOS)
            h2 = img2.size[1]
        
        # 创建新图片（垂直拼接）
        merged = Image.new('RGB', (max_width, h1 + h2))
        merged.paste(img1, (0, 0))
        merged.paste(img2, (0, h1))
        
        return merged, h1
        
    except Exception as e:
        return None, None


def process_single_image(args):
    """
    处理单张图片的合并（用于多进程）
    
    Args:
        args: (img1_path, img2_path, output_path, label1, label2)
        
    Returns:
        success: bool
    """
    img1_path, img2_path, output_path, label1, label2 = args
    
    # 检查第二个目录是否存在对应图片
    if not img2_path.exists():
        return False
    
    # 合并图片
    merged_img, mid_y = merge_images_vertical(img1_path, img2_path)
    
    if merged_img is not None:
        # 添加标签和分界线
        draw = ImageDraw.Draw(merged_img)
        
        # 在分界线位置画一条白线
        w = merged_img.size[0]
        draw.line([(0, mid_y), (w, mid_y)], fill='white', width=3)
        
        # 添加文字标签
        try:
            # 尝试使用系统字体
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            # 如果找不到字体，使用默认字体
            font = ImageFont.load_default()
        
        draw.text((10, 10), label1, fill='white', font=font)
        draw.text((10, mid_y + 10), label2, fill='white', font=font)
        
        # 保存
        merged_img.save(str(output_path))
        return True
    
    return False


def collect_all_tasks(dir1, dir2, output_dir, label1, label2):
    """
    收集所有待处理的图片任务
    
    Returns:
        list of (img1_path, img2_path, output_path, label1, label2)
    """
    tasks = []
    subfolders1 = sorted([d for d in dir1.iterdir() if d.is_dir()])
    
    for subfolder1 in subfolders1:
        subfolder_name = subfolder1.name
        subfolder2 = dir2 / subfolder_name
        output_subfolder = output_dir / subfolder_name
        
        # 检查第二个目录是否存在对应文件夹
        if not subfolder2.exists():
            continue
        
        # 创建输出子文件夹
        output_subfolder.mkdir(parents=True, exist_ok=True)
        
        # 获取第一个文件夹中的所有图片
        image_files1 = sorted(list(subfolder1.glob("*.png")) + 
                            list(subfolder1.glob("*.jpg")) + 
                            list(subfolder1.glob("*.jpeg")))
        
        # 添加任务
        for img1_path in image_files1:
            img_name = img1_path.name
            img2_path = subfolder2 / img_name
            output_path = output_subfolder / img_name
            tasks.append((img1_path, img2_path, output_path, label1, label2))
    
    return tasks


def merge_visualization_dirs(dir1, dir2, output_dir, label1="Model 1", label2="Model 2", num_workers=None):
    """
    合并两个可视化目录（多进程版本）
    
    Args:
        dir1: 第一个目录路径
        dir2: 第二个目录路径
        output_dir: 输出目录路径
        label1: 第一个模型的标签
        label2: 第二个模型的标签
        num_workers: 进程数（默认为CPU核心数）
    """
    dir1 = Path(dir1)
    dir2 = Path(dir2)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"目录1: {dir1}")
    print(f"目录2: {dir2}")
    print(f"输出目录: {output_dir}")
    print(f"使用 {num_workers} 个进程并行处理")
    
    # 收集所有任务
    print("正在扫描文件...")
    tasks = collect_all_tasks(dir1, dir2, output_dir, label1, label2)
    print(f"找到 {len(tasks)} 张图片需要处理")
    
    # 使用多进程处理
    total_merged = 0
    total_skipped = 0
    
    with Pool(num_workers) as pool:
        # 使用imap_unordered以支持进度条，并设置较大的chunksize以减少进程间通信开销
        results = list(tqdm(
            pool.imap_unordered(process_single_image, tasks, chunksize=10),
            total=len(tasks),
            desc="合并图片"
        ))
    
    # 统计结果
    total_merged = sum(results)
    total_skipped = len(results) - total_merged
    
    print(f"\n完成！")
    print(f"成功合并: {total_merged} 张图片")
    print(f"跳过: {total_skipped} 张图片")


def main():
    parser = argparse.ArgumentParser(description="合并两个可视化目录的图片（多进程加速版）")
    parser.add_argument("--dir1", type=str, 
                       default="work_dirs/cgnet_ep24_fp16/vis_pred",
                       help="第一个可视化目录路径")
    parser.add_argument("--dir2", type=str,
                       default="work_dirs/cgnet_ep24_bit_diffusion_no_geo_smooth_0.3720/vis_pred",
                       help="第二个可视化目录路径")
    parser.add_argument("--output", type=str,
                       default="work_dirs/merged_visualizations",
                       help="输出目录路径")
    parser.add_argument("--label1", type=str,
                       default="Baseline (FP16)",
                       help="第一个模型的标签")
    parser.add_argument("--label2", type=str,
                       default="Bit Diffusion (No Geo + Smooth)",
                       help="第二个模型的标签")
    parser.add_argument("--num-workers", type=int,
                       default=32,
                       help="并行进程数（默认为CPU核心数）")
    
    args = parser.parse_args()
    
    merge_visualization_dirs(
        args.dir1,
        args.dir2,
        args.output,
        args.label1,
        args.label2,
        args.num_workers
    )


if __name__ == "__main__":
    main()
