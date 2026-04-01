# Copyright (C) 2024 Xiaomi Corporation.

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, 
# software distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and limitations under the License.

import argparse
import mmcv
import os
import torch
import warnings

import sys
dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dir_path)

from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel, collate
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet3d.utils import collect_env, get_root_logger
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
import os.path as osp
import numpy as np
import cv2
import networkx as nx
from PIL import Image as PILImage

TOPOLOGY_EDGE_COLOR = (0, 165, 255)  # BGR orange
ARROW_TIP_SCALE = 14
LANE_LINE_THICKNESS = 3
TOPOLOGY_LINE_THICKNESS = 5

def parse_args():
    parser = argparse.ArgumentParser(description='vis map gt and pred')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--samples', default=2000, help='samples to visualize')
    parser.add_argument(
        '--example-index',
        type=int,
        default=None,
        help='only visualize one dataset example by dataloader index (0-based)')
    parser.add_argument(
        '--example-token',
        type=str,
        default=None,
        help='only visualize one dataset example by scene_token (exact match)')
    parser.add_argument(
        '--show-dir', help='directory where visualizations will be saved')
    parser.add_argument('--save-video', action='store_true', help='generate video')
    parser.add_argument('--save-gif', action='store_true',
                        help='save diffusion denoising process as GIF (like generate.py visualize_all_iter)')
    parser.add_argument('--gif-fps', type=float, default=3,
                        help='GIF playback speed in frames per second (default: 3)')
    parser.add_argument('--gif-max-steps', type=int, default=None,
                        help='max diffusion steps to render into GIF (uniformly sampled, include first/last)')
    parser.add_argument(
        '--gt-format',
        type=str,
        nargs='+',
        default=['se_points',],
        help='vis format, default should be "points",'
        'support ["se_pts","bbox","fixed_num_pts","polyline_pts"]')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.show_dir is None:
        # 优先使用 checkpoint 路径的父目录名（包含评估指标）
        # 例如：cgnet_ep24_bit_diffusion_no_geo_smooth_0.3720
        checkpoint_dir = osp.dirname(args.checkpoint)
        if 'work_dirs' in checkpoint_dir:
            # 使用 checkpoint 所在目录的名字
            work_dir_name = osp.basename(checkpoint_dir)
            args.show_dir = osp.join(checkpoint_dir, 'vis_pred')
        else:
            # Fallback：使用配置文件名
            args.show_dir = osp.join('./work_dirs', 
                                    osp.splitext(osp.basename(args.config))[0],
                                    'vis_pred')
    # create vis_label dir
    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    cfg.dump(osp.join(args.show_dir, osp.basename(args.config)))
    logger = get_root_logger()
    logger.info(f'DONE create vis_pred dir: {args.show_dir}')


    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True #TODO, this is a hack
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        # workers_per_gpu=cfg.data.workers_per_gpu,
        workers_per_gpu=4,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    logger.info('Done build test data set')

    # build the model and load checkpoint
    # import pdb;pdb.set_trace()
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.pts_bbox_head.bbox_coder.vis_mode = True
    model.pts_bbox_head.bbox_coder.score_threshold = 0.4
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    logger.info('loading check point')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    logger.info('DONE load check point')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    topo_vis_threshold = getattr(model.module.pts_bbox_head.bbox_coder, 'adj_threshold', 0.5)

    logger.info('BEGIN vis test dataset samples gt label & pred')


    dataset = data_loader.dataset
    out_dir_list = []
    num_visualized = 0
    target_index = args.example_index
    target_token = args.example_token
    if target_index is not None and target_index < 0:
        raise ValueError('--example-index must be >= 0')
    if target_index is not None and target_token is not None:
        raise ValueError('Please specify only one of --example-index or --example-token')

    def _get_scene_token_from_info(info):
        if 'scene_token' in info:
            return info['scene_token']
        if 'log_id' in info:
            return info['log_id']
        return None

    # Fast path: resolve one sample directly instead of sequential dataloader scanning.
    if target_token is not None:
        matched_indices = []
        for idx, info in enumerate(dataset.data_infos):
            if _get_scene_token_from_info(info) == target_token:
                matched_indices.append(idx)

        if len(matched_indices) == 0:
            logger.warning(f'No sample found for --example-token={target_token}.')
            return

        if len(matched_indices) > 1:
            logger.warning(f'Found {len(matched_indices)} samples for --example-token={target_token}; '
                           f'use the first one (index={matched_indices[0]}).')
        target_index = matched_indices[0]

    if target_index is not None:
        if target_index >= len(dataset):
            logger.warning(f'No sample visualized for --example-index={target_index}. '
                           f'It is out of range [0, {len(dataset) - 1}].')
            return

        logger.info(f'Preparing single sample: index={target_index}')
        data = collate([dataset[target_index]], samples_per_gpu=1)
        prog_bar = mmcv.ProgressBar(1)
        iter_data = [(target_index, data)]
    else:
        prog_bar = mmcv.ProgressBar(len(dataset))
        iter_data = enumerate(data_loader)

    # import pdb;pdb.set_trace()
    for i, data in iter_data:

        img_metas = data['img_metas'][0].data[0]
        token = img_metas[0].get('scene_token', None)
        if target_token is not None and token != target_token:
            prog_bar.update()
            continue

        if ~(data['gt_labels_3d'].data[0][0] != -1).any():
            # import pdb;pdb.set_trace()
            logger.error(f'\n empty gt for index {i}, continue')
            prog_bar.update()  
            if target_index is not None or target_token is not None:
                break
            continue
        with torch.no_grad():
            logger.info(f'Running model forward for sample index={i} ...')
            result = model(return_loss=False, rescale=True, **data)
        
        # import pdb;pdb.set_trace()
        token = img_metas[0]['scene_token']
        gt_bboxes_3d = data['gt_bboxes_3d'].data[0]
        gt_labels_3d = data['gt_labels_3d'].data[0]

        gt_lines_fixed_num_pts = (gt_bboxes_3d[0].fixed_num_sampled_points)
        gt_graph = gt_bboxes_3d[0].get_graph_gt_vis
        img = img_metas[0]['ori_image']

        result_dic = result[0]['pts_bbox']
        line = np.array([l.cpu().numpy() for l in result_dic['line']])
        line_scores = result_dic['line_scores'].cpu().numpy()
        line_labels = result_dic['line_labels'].cpu().numpy()

        pts = result_dic['pts'].cpu().numpy()
        pts_scores = result_dic['pts_scores'].cpu().numpy()
        pts_labels = result_dic['pts_labels'].cpu().numpy()
        graph = result_dic['graph']
        
        if gt_labels_3d[0].shape[0] != gt_lines_fixed_num_pts.shape[0]:
            gt_labels_3d = np.zeros(gt_lines_fixed_num_pts.shape[0])
        else:
            gt_labels_3d = gt_labels_3d[0].cpu().numpy()

        gts = [{'pts':gt_lines_fixed_num_pts[i,...].cpu().numpy(), 'type':gt_labels_3d[i]} for i in range(gt_lines_fixed_num_pts.shape[0])]
        line_preds = [{'line':line[i,...], 'type':line_labels[i],'confidence_level':line_scores[i]} for i in range(line.shape[0])]
        pts_preds = [{'pts':pts[i,...], 'type':pts_labels[i],'confidence_level':pts_scores[i]} for i in range(pts.shape[0])]
        topo_graph = build_topology_overlay_graph(line, result_dic.get('adj_matrix', None), topo_vis_threshold)
        out_dir = os.path.join(args.show_dir, token)
        if out_dir not in out_dir_list:
            out_dir_list.append(out_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_path = os.path.join(out_dir, "{}.jpg".format(i))
        render(img, out_path, gts, graph, gt_graph, topo_graph=topo_graph)

        # ====================================================
        # GIF 动画：可视化扩散去噪过程（类似 generate.py 的 viz_gif）
        # ====================================================
        # 核心思路（参考 generate.py 的 edm_sampler + visualize_all_iter）：
        #   1. edm_sampler 在每个去噪步收集中间态 inter_results
        #   2. visualize_all_iter 将每步的多边形渲染为帧，合成 GIF
        # 对应到 CGNet（BitDiffusion）：
        #   1. inference_sampling_with_intermediates 收集每步二值化邻接矩阵
        #   2. 对每步邻接矩阵，用已有的 pts 重建拓扑图并渲染
        # ====================================================
        if args.save_gif:
            head = model.module.pts_bbox_head

            # 检查是否有缓存的条件特征（forward 推理时写入 head._last_cond）
            if not (hasattr(head, '_last_cond') and head._last_cond is not None):
                logger.warning(f'[GIF] Sample {i}: head._last_cond is None, skip. '
                               '(Model may not be CGTopoHeadBitDiffusion)')
                prog_bar.update()
                if target_index is not None or target_token is not None:
                    break
                continue

            query_index = result_dic.get('query_index', None)
            if query_index is None or len(query_index) == 0:
                logger.warning(f'[GIF] Sample {i}: no detected queries, skip.')
                prog_bar.update()
                if target_index is not None or target_token is not None:
                    break
                continue

            # 取 batch=1 的 cond（head._last_cond shape: [B, N, N, cond_dim]）
            cond_single = head._last_cond[:1]  # [1, N, N, cond_dim]
            q_idx = query_index.cpu().numpy()   # [M] detected query indices

            # 运行带中间步骤的采样（不影响已有的最终预测结果）
            logger.info(f'[GIF] Sampling diffusion intermediates for sample index={i} ...')
            intermediates = head.inference_sampling_with_intermediates(cond_single)
            # intermediates: list of (num_inference_steps+1) arrays, each [N, N]

            if args.gif_max_steps is not None:
                if args.gif_max_steps <= 0:
                    raise ValueError('--gif-max-steps must be > 0')
                if len(intermediates) > args.gif_max_steps + 1:
                    keep_idx = np.linspace(0, len(intermediates) - 1,
                                           num=args.gif_max_steps + 1,
                                           dtype=int)
                    intermediates = [intermediates[k] for k in keep_idx]
                    logger.info(f'[GIF] Downsample intermediates to {len(intermediates)} frames '
                                f'via --gif-max-steps={args.gif_max_steps}.')

            # line: numpy array [M, num_pts, 2] in physical coords
            line_list = [line[j] for j in range(len(line))]  # list of [num_pts, 2]

            gif_frames = []
            total_steps = len(intermediates) - 1  # 去噪步数（不含初始噪声）

            for step_k, adj_full in enumerate(intermediates):
                # 提取检测到的 query 对应的子邻接矩阵
                adj_sub = adj_full[q_idx][:, q_idx]         # [M, M]
                am = (adj_sub > head.inference_threshold).astype(np.float32)
                np.fill_diagonal(am, 0)                     # 去除自环

                # 用当前步的邻接矩阵重建图
                G_step = build_step_graph(line_list, am)

                # 渲染本帧：相机图像 + GT图 + 当前预测图
                step_label = 'noise' if step_k == 0 else f'{step_k}/{total_steps}'
                frame = render_to_array(img, G_step, gt_graph, step_label)
                gif_frames.append(frame)

            # 保存 GIF
            gif_path = os.path.join(out_dir, '{}_diffusion.gif'.format(i))
            logger.info(f'[GIF] Rendering and saving {len(gif_frames)} frames ...')
            save_as_gif(gif_frames, gif_path, fps=args.gif_fps)
            logger.info(f'[GIF] Saved {len(gif_frames)}-frame GIF → {gif_path}')

        prog_bar.update()
        num_visualized += 1

        if target_index is not None or target_token is not None:
            break
        if num_visualized >= int(args.samples):
            logger.info(f'Reach --samples limit ({args.samples}), stop visualization.')
            break

    if target_index is not None and num_visualized == 0:
        logger.warning(f'No sample visualized for --example-index={target_index}. '
                       'It may be out of range or filtered as empty GT.')
    if target_token is not None and num_visualized == 0:
        logger.warning(f'No sample visualized for --example-token={target_token}. '
                       'It may not exist in the test split or is filtered as empty GT.')

    # # create video for scenes
    if args.save_video: 
        logger.info('\n Creat Video')
        for path in out_dir_list:
            creat_video(path)

    logger.info('\n DONE vis test dataset samples gt label & pred')


def build_step_graph(pts_list, adj_matrix):
    """
    根据检测到的车道线和当前步的二值邻接矩阵，构建 NetworkX 有向图。
    
    类比 generate.py 中每步 inter_results 对应的多边形坐标：
    这里的"多边形坐标"是固定的 pts，变化的是拓扑连接关系（adj_matrix）。
    
    Args:
        pts_list: List of [num_pts, 2] arrays - 检测到的车道线点集（物理坐标）
        adj_matrix: [M, M] float array - 当前扩散步的二值邻接矩阵
        
    Returns:
        G: nx.DiGraph - 包含车道内部边和跨车道拓扑边的有向图
    """
    M = len(pts_list)
    if M == 0:
        return nx.DiGraph()

    instance_points = np.concatenate([p for p in pts_list], axis=0)  # [total_pts, 2]
    line_length = [len(p) for p in pts_list]
    nums_pts = len(instance_points)

    G = nx.DiGraph()
    for k in range(nums_pts):
        G.add_node(k, pos=instance_points[k])

    start_loc = [0] + list(np.cumsum(line_length)[:-1])

    # 车道线内部边（沿线方向连接）
    for i in range(M):
        idx = start_loc[i] + np.arange(line_length[i])
        for j in range(1, len(idx)):
            G.add_edge(int(idx[j - 1]), int(idx[j]), edge_type='lane')

    # 跨车道拓扑边（由邻接矩阵决定）
    for i in range(M):
        for j in range(M):
            if adj_matrix[i, j] > 0.5 and i != j:
                end_i = start_loc[i] + line_length[i] - 1   # 车道 i 的末尾点
                start_j = start_loc[j]                       # 车道 j 的起始点
                G.add_edge(int(end_i), int(start_j), edge_type='topology')
    return G


def build_topology_overlay_graph(lines, adj_scores, threshold=0.5):
    """Build topology-only graph (end->start links) for static overlay rendering."""
    G = nx.DiGraph()
    if adj_scores is None:
        return G

    adj = np.asarray(adj_scores)
    if adj.ndim != 2:
        return G

    m = min(len(lines), adj.shape[0], adj.shape[1])
    if m == 0:
        return G

    for i in range(m):
        start_node = f's{i}'
        end_node = f'e{i}'
        G.add_node(start_node, pos=np.asarray(lines[i][0]))
        G.add_node(end_node, pos=np.asarray(lines[i][-1]))

    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            if adj[i, j] > threshold:
                G.add_edge(f'e{i}', f's{j}', edge_type='topology')
    return G


def render_to_array(imgs, graph, gt_graph, step_label='', height=900):
    """
    类比 generate.py 的 visualize_all_iter：将某一扩散步的拓扑状态渲染为 numpy 图像。
    
    布局：[相机拼图 | GT 图 | 当前步预测图]
    
    Args:
        imgs:       6 张相机图像 (f, fr, fl, b, bl, br)
        graph:      当前步的预测拓扑图（nx.DiGraph）
        gt_graph:   GT 拓扑图（nx.DiGraph），静态不变
        step_label: 步骤标签字符串（显示在右上角）
        height:     BEV 图的高度（像素）
        
    Returns:
        canvas: numpy array [H, W, 3] BGR
    """
    COLOR = ((0, 0, 0), (116, 92, 75), (83, 97, 96), (255, 0, 0), (0, 0, 255))
    scale = height // 30
    map_img_p  = np.ones((height * 2, height, 3), dtype=np.uint8) * 255
    map_img_gt = np.ones((height * 2, height, 3), dtype=np.uint8) * 255

    def draw_graph(canvas, g, node_color, lane_edge_color, topology_edge_color=None):
        for e in g.edges():
            x1 = g.nodes[e[0]]['pos'][0] * scale + height // 2
            y1 = height - g.nodes[e[0]]['pos'][1] * scale
            x2 = g.nodes[e[1]]['pos'][0] * scale + height // 2
            y2 = height - g.nodes[e[1]]['pos'][1] * scale
            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            edge_attr = g.edges[e]
            edge_color = lane_edge_color
            thickness = LANE_LINE_THICKNESS
            if topology_edge_color is not None and edge_attr.get('edge_type', 'lane') == 'topology':
                edge_color = topology_edge_color
                thickness = TOPOLOGY_LINE_THICKNESS
            cv2.circle(canvas, (int(x1), int(y1)), 8, node_color, -1)
            cv2.circle(canvas, (int(x2), int(y2)), 8, node_color, -1)
            cv2.arrowedLine(canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                            edge_color, thickness, tipLength=float(ARROW_TIP_SCALE / (dist + 1)))

    if gt_graph is not None:
        draw_graph(map_img_gt, gt_graph, COLOR[-1], COLOR[0])
    draw_graph(map_img_p, graph, COLOR[-1], COLOR[3], TOPOLOGY_EDGE_COLOR)

    # 在预测图右上角显示当前步骤
    if step_label:
        cv2.putText(map_img_p, f'Step: {step_label}',
                    (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 120, 0), 2)

    f, fr, fl, b, bl, br = imgs
    cam_canvas = cv2.vconcat([cv2.hconcat([fl, f, fr]),
                              cv2.hconcat([bl, b, br])])

    # 统一类型为 uint8（相机图像可能是 float32，与地图 uint8 类型不同会导致 hconcat 断言失败）
    cam_canvas = np.clip(cam_canvas, 0, 255).astype(np.uint8)

    # 统一高度后拼接（cv2.resize dsize=(width, height)）
    target_h = cam_canvas.shape[0]
    map_gt_r   = cv2.resize(map_img_gt.astype(np.uint8), (height, target_h))
    map_pred_r = cv2.resize(map_img_p.astype(np.uint8),  (height, target_h))
    canvas = cv2.hconcat([cam_canvas, map_gt_r, map_pred_r])
    return canvas  # BGR uint8 numpy


def save_as_gif(frames_bgr, gif_path, fps=3):
    """
    类比 generate.py 的 visualize_all_iter：将多帧 BGR numpy 图像保存为 GIF 文件。
    
    使用 PIL 的 save(save_all=True, append_images=...) 接口，
    与 generate.py 调用 imageio/PIL 创建 GIF 的方式相同。
    
    Args:
        frames_bgr: list of [H, W, 3] BGR numpy arrays
        gif_path:   输出 GIF 文件路径
        fps:        每秒帧数（default: 3）
    """
    if not frames_bgr:
        return

    duration_ms = int(1000 / max(fps, 0.1))

    pil_frames = []
    for frame in frames_bgr:
        # BGR → RGB，再转 PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 限制最大宽度（GIF 文件体积控制）
        h, w = frame_rgb.shape[:2]
        if w > 1400:
            scale_f = 1400 / w
            frame_rgb = cv2.resize(frame_rgb, (1400, int(h * scale_f)))
        pil_frames.append(PILImage.fromarray(frame_rgb))

    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,          # 无限循环
        optimize=False,  # 不压缩，保留颜色质量
    )


def render(imgs, out_file, gt, graph, gt_graph, topo_graph=None, height=900):
        

        COLOR = ((0, 0, 0), (116, 92, 75), (83, 97, 96), (255, 0, 0), (0, 0, 255))

        scale = height // 30
        map_img_p = np.ones((height*2, height, 3), dtype=np.uint8) * 255
        map_img_gt = np.ones((height*2, height, 3), dtype=np.uint8) * 255

        if gt is not None:
            for e in gt_graph.edges():
                x1 = gt_graph.nodes[e[0]]['pos'][0] *scale
                y1 = gt_graph.nodes[e[0]]['pos'][1] *scale
                x2 = gt_graph.nodes[e[1]]['pos'][0] *scale
                y2 = gt_graph.nodes[e[1]]['pos'][1] *scale
                x1 += height//2
                y1 = height - y1
                x2 += height//2
                y2 = height - y2
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                cv2.circle(map_img_gt, (np.int32(x1), np.int32(y1)), 8, COLOR[-1], -1)
                cv2.circle(map_img_gt, (np.int32(x2), np.int32(y2)), 8, COLOR[-1], -1)
                cv2.arrowedLine(map_img_gt, 
                        (np.int32(x1), np.int32(y1)), 
                        (np.int32(x2), np.int32(y2)), 
                        color = COLOR[0], 
                    thickness=LANE_LINE_THICKNESS,
                    tipLength=float(ARROW_TIP_SCALE/(distance+1)))
                
        for e in graph.edges():
            x1 = graph.nodes[e[0]]['pos'][0] *scale
            y1 = graph.nodes[e[0]]['pos'][1] *scale
            x2 = graph.nodes[e[1]]['pos'][0] *scale
            y2 = graph.nodes[e[1]]['pos'][1] *scale
            x1 += height//2
            y1 = height - y1
            x2 += height//2
            y2 = height - y2
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            cv2.circle(map_img_p, (np.int32(x1), np.int32(y1)), 8, COLOR[-1], -1)
            cv2.circle(map_img_p, (np.int32(x2), np.int32(y2)), 8, COLOR[-1], -1)
            cv2.arrowedLine(map_img_p, 
                    (np.int32(x1), np.int32(y1)), 
                    (np.int32(x2), np.int32(y2)), 
                    color = COLOR[3], 
                    thickness=LANE_LINE_THICKNESS,
                    tipLength=float(ARROW_TIP_SCALE/(distance+1)))

        if topo_graph is not None:
            for e in topo_graph.edges():
                x1 = topo_graph.nodes[e[0]]['pos'][0] * scale
                y1 = topo_graph.nodes[e[0]]['pos'][1] * scale
                x2 = topo_graph.nodes[e[1]]['pos'][0] * scale
                y2 = topo_graph.nodes[e[1]]['pos'][1] * scale
                x1 += height // 2
                y1 = height - y1
                x2 += height // 2
                y2 = height - y2
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                cv2.arrowedLine(
                    map_img_p,
                    (np.int32(x1), np.int32(y1)),
                    (np.int32(x2), np.int32(y2)),
                    color=TOPOLOGY_EDGE_COLOR,
                    thickness=TOPOLOGY_LINE_THICKNESS,
                    tipLength=float(ARROW_TIP_SCALE / (distance + 1)))


        f, fr, fl, b, bl, br = imgs
        canvas = cv2.vconcat([cv2.hconcat([fl, f, fr]), cv2.hconcat([bl, b, br])])
        canvas = cv2.hconcat([canvas, map_img_gt.astype(np.float32), map_img_p.astype(np.float32), ])
        cv2.imwrite(out_file, canvas)

def creat_video(folder_path):
    video_name = folder_path + '/output.mp4'

    file_names = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    file_names.sort(key=lambda x:int(x.split('.')[0]))
    
    img = cv2.imread(os.path.join(folder_path, file_names[0]))
    height, width, channels = img.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_name, fourcc, 5, (width, height))

    for file_name in file_names:
        img = cv2.imread(os.path.join(folder_path, file_name))
        video_writer.write(img)

    video_writer.release()


if __name__ == '__main__':
    main()
