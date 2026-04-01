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
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
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
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='vis map gt and pred')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--samples', default=2000, type=int, help='samples to visualize')
    parser.add_argument('--sample-idx', default=None, type=int, help='specific sample index to visualize')
    parser.add_argument(
        '--show-dir', help='directory where visualizations will be saved')
    parser.add_argument('--save-video', action='store_true', help='generate video')
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
        # 从 checkpoint 路径中提取工作目录
        checkpoint_dir = osp.dirname(args.checkpoint)
        if 'work_dirs' in checkpoint_dir:
            # checkpoint 在 work_dirs 下，使用其父目录
            args.show_dir = osp.join(checkpoint_dir, 'vis_pred')
        else:
            # 降级到使用配置文件名
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

    logger.info('BEGIN vis test dataset samples gt label & pred')


    dataset = data_loader.dataset
    out_dir_list = []
    
    # 如果指定了 sample-idx，直接获取该样本
    if args.sample_idx is not None:
        logger.info(f'Visualizing only sample index: {args.sample_idx}')
        data = dataset[args.sample_idx]
        data = data_loader.collate_fn([data])
        
        if ~(data['gt_labels_3d'].data[0][0] != -1).any():
            logger.error(f'\n empty gt for index {args.sample_idx}')
            return
            
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        
        i = args.sample_idx
        # 处理结果...
        img_metas = data['img_metas'][0].data[0]
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
        
        # ========== 添加调试信息 ==========
        print("\n" + "="*80)
        print(f"DEBUG INFO for Sample {args.sample_idx}")
        print("="*80)
        
        print("\n[1] Ground Truth Data:")
        print(f"  - gt_lines_fixed_num_pts.shape: {gt_lines_fixed_num_pts.shape}")
        print(f"  - Number of GT lanes: {gt_lines_fixed_num_pts.shape[0]}")
        print(f"  - Points per lane: {gt_lines_fixed_num_pts.shape[1]}")
        print(f"  - Coordinate dimension: {gt_lines_fixed_num_pts.shape[2]}")
        print(f"  - Sample points from first GT lane:")
        print(f"    First point:  {gt_lines_fixed_num_pts[0, 0].cpu().numpy()}")
        print(f"    Middle point: {gt_lines_fixed_num_pts[0, 10].cpu().numpy()}")
        print(f"    Last point:   {gt_lines_fixed_num_pts[0, -1].cpu().numpy()}")
        print(f"  - Coordinate range X: [{gt_lines_fixed_num_pts[..., 0].min():.2f}, {gt_lines_fixed_num_pts[..., 0].max():.2f}]")
        print(f"  - Coordinate range Y: [{gt_lines_fixed_num_pts[..., 1].min():.2f}, {gt_lines_fixed_num_pts[..., 1].max():.2f}]")
        
        print(f"\n[2] Ground Truth Graph (gt_graph):")
        print(f"  - Type: {type(gt_graph)}")
        print(f"  - Number of nodes: {gt_graph.number_of_nodes()}")
        print(f"  - Number of edges: {gt_graph.number_of_edges()}")
        print(f"  - Nodes: {list(gt_graph.nodes())}")
        print(f"  - Edges (connections): {list(gt_graph.edges())[:10]}")  # 只显示前10条边
        if gt_graph.number_of_nodes() > 0:
            first_node = list(gt_graph.nodes())[0]
            print(f"  - Node {first_node} attributes: {gt_graph.nodes[first_node]}")
            print(f"  - Node {first_node} position: {gt_graph.nodes[first_node]['pos']}")
        
        print(f"\n[3] Prediction Data:")
        print(f"  - line.shape: {line.shape}")
        print(f"  - Number of predicted lanes: {line.shape[0]}")
        print(f"  - Points per predicted lane: {line.shape[1]}")
        print(f"  - line_scores.shape: {line_scores.shape}")
        print(f"  - line_labels.shape: {line_labels.shape}")
        print(f"  - Sample from first predicted lane:")
        print(f"    First point:  {line[0, 0]}")
        print(f"    Middle point: {line[0, 10]}")
        print(f"    Last point:   {line[0, -1]}")
        print(f"    Confidence:   {line_scores[0]:.4f}")
        print(f"    Label:        {line_labels[0]}")
        
        print(f"\n[4] Prediction Graph (graph):")
        print(f"  - Type: {type(graph)}")
        print(f"  - Number of nodes: {graph.number_of_nodes()}")
        print(f"  - Number of edges: {graph.number_of_edges()}")
        print(f"  - Nodes: {list(graph.nodes())}")
        print(f"  - Edges (predicted connections): {list(graph.edges())[:10]}")  # 只显示前10条边
        if graph.number_of_edges() > 0:
            first_edge = list(graph.edges())[0]
            print(f"  - First edge: {first_edge}")
            print(f"    Node {first_edge[0]} pos: {graph.nodes[first_edge[0]]['pos']}")
            print(f"    Node {first_edge[1]} pos: {graph.nodes[first_edge[1]]['pos']}")
        
        print(f"\n[5] Processed Data:")
        print(f"  - len(gts): {len(gts)}")
        print(f"  - len(line_preds): {len(line_preds)}")
        print(f"  - gts[0] keys: {gts[0].keys()}")
        print(f"  - gts[0]['pts'].shape: {gts[0]['pts'].shape}")
        print(f"  - line_preds[0] keys: {line_preds[0].keys()}")
        print(f"  - line_preds[0]['line'].shape: {line_preds[0]['line'].shape}")
        
        print(f"\n[6] High Confidence Predictions:")
        high_conf_mask = line_scores > 0.4
        print(f"  - Predictions with score > 0.4: {high_conf_mask.sum()}")
        if high_conf_mask.sum() > 0:
            high_conf_scores = line_scores[high_conf_mask]
            print(f"  - Their scores: {high_conf_scores[:10]}")  # 显示前10个
        
        print("\n" + "="*80 + "\n")
        # ========== 调试信息结束 ==========
        
        out_dir = os.path.join(args.show_dir, token)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "{}.jpg".format(args.sample_idx))
        render(img, out_path, gts, graph, gt_graph, line_preds)
        logger.info(f'Done visualizing sample {args.sample_idx}')
        return
    
    # 否则遍历所有样本
    prog_bar = mmcv.ProgressBar(len(dataset))
    # import pdb;pdb.set_trace()
    for i, data in enumerate(data_loader):
        if ~(data['gt_labels_3d'].data[0][0] != -1).any():
            # import pdb;pdb.set_trace()
            logger.error(f'\n empty gt for index {i}, continue')
            prog_bar.update()  
            continue
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        
        # import pdb;pdb.set_trace()
        img_metas = data['img_metas'][0].data[0]
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
        
        # ========== 添加调试信息 ==========
        print("\n" + "="*80)
        print(f"DEBUG INFO for Sample {i}")
        print("="*80)
        
        print("\n[1] Ground Truth Data:")
        print(f"  - gt_lines_fixed_num_pts.shape: {gt_lines_fixed_num_pts.shape}")
        print(f"  - Number of GT lanes: {gt_lines_fixed_num_pts.shape[0]}")
        print(f"  - Points per lane: {gt_lines_fixed_num_pts.shape[1]}")
        print(f"  - Coordinate dimension: {gt_lines_fixed_num_pts.shape[2]}")
        print(f"  - Sample points from first GT lane:")
        print(f"    First point:  {gt_lines_fixed_num_pts[0, 0].cpu().numpy()}")
        print(f"    Middle point: {gt_lines_fixed_num_pts[0, 10].cpu().numpy()}")
        print(f"    Last point:   {gt_lines_fixed_num_pts[0, -1].cpu().numpy()}")
        print(f"  - Coordinate range X: [{gt_lines_fixed_num_pts[..., 0].min():.2f}, {gt_lines_fixed_num_pts[..., 0].max():.2f}]")
        print(f"  - Coordinate range Y: [{gt_lines_fixed_num_pts[..., 1].min():.2f}, {gt_lines_fixed_num_pts[..., 1].max():.2f}]")
        
        print(f"\n[2] Ground Truth Graph (gt_graph):")
        print(f"  - Type: {type(gt_graph)}")
        print(f"  - Number of nodes: {gt_graph.number_of_nodes()}")
        print(f"  - Number of edges: {gt_graph.number_of_edges()}")
        print(f"  - Nodes: {list(gt_graph.nodes())}")
        print(f"  - Edges (connections): {list(gt_graph.edges())[:10]}")  # 只显示前10条边
        if gt_graph.number_of_nodes() > 0:
            first_node = list(gt_graph.nodes())[0]
            print(f"  - Node {first_node} attributes: {gt_graph.nodes[first_node]}")
            print(f"  - Node {first_node} position: {gt_graph.nodes[first_node]['pos']}")
        
        print(f"\n[3] Prediction Data:")
        print(f"  - line.shape: {line.shape}")
        print(f"  - Number of predicted lanes: {line.shape[0]}")
        print(f"  - Points per predicted lane: {line.shape[1]}")
        print(f"  - line_scores.shape: {line_scores.shape}")
        print(f"  - line_labels.shape: {line_labels.shape}")
        print(f"  - Sample from first predicted lane:")
        print(f"    First point:  {line[0, 0]}")
        print(f"    Middle point: {line[0, 10]}")
        print(f"    Last point:   {line[0, -1]}")
        print(f"    Confidence:   {line_scores[0]:.4f}")
        print(f"    Label:        {line_labels[0]}")
        
        print(f"\n[4] Prediction Graph (graph):")
        print(f"  - Type: {type(graph)}")
        print(f"  - Number of nodes: {graph.number_of_nodes()}")
        print(f"  - Number of edges: {graph.number_of_edges()}")
        print(f"  - Nodes: {list(graph.nodes())}")
        print(f"  - Edges (predicted connections): {list(graph.edges())[:10]}")  # 只显示前10条边
        if graph.number_of_edges() > 0:
            first_edge = list(graph.edges())[0]
            print(f"  - First edge: {first_edge}")
            print(f"    Node {first_edge[0]} pos: {graph.nodes[first_edge[0]]['pos']}")
            print(f"    Node {first_edge[1]} pos: {graph.nodes[first_edge[1]]['pos']}")
        
        print(f"\n[5] Processed Data:")
        print(f"  - len(gts): {len(gts)}")
        print(f"  - len(line_preds): {len(line_preds)}")
        print(f"  - gts[0] keys: {gts[0].keys()}")
        print(f"  - gts[0]['pts'].shape: {gts[0]['pts'].shape}")
        print(f"  - line_preds[0] keys: {line_preds[0].keys()}")
        print(f"  - line_preds[0]['line'].shape: {line_preds[0]['line'].shape}")
        
        print(f"\n[6] High Confidence Predictions:")
        high_conf_mask = line_scores > 0.4
        print(f"  - Predictions with score > 0.4: {high_conf_mask.sum()}")
        if high_conf_mask.sum() > 0:
            high_conf_scores = line_scores[high_conf_mask]
            print(f"  - Their scores: {high_conf_scores[:10]}")  # 显示前10个
        
        print("\n" + "="*80 + "\n")
        # ========== 调试信息结束 ==========
        
        out_dir = os.path.join(args.show_dir, token)
        if out_dir not in out_dir_list:
            out_dir_list.append(out_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_path = os.path.join(out_dir, "{}.jpg".format(i))
        render(img, out_path, gts, graph, gt_graph, line_preds)

        prog_bar.update()

    # # create video for scenes
    if args.save_video: 
        logger.info('\n Creat Video')
        for path in out_dir_list:
            creat_video(path)

    logger.info('\n DONE vis test dataset samples gt label & pred')


def render(imgs, out_file, gt, graph, gt_graph, line_preds, height=900):
        

        COLOR = ((0, 0, 0), (116, 92, 75), (83, 97, 96), (255, 0, 0), (0, 0, 255))
        LINE_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

        scale = height // 30
        
        # 预测结果的4张图
        map_img_p = np.ones((height*2, height, 3), dtype=np.uint8) * 255
        map_img_p_no_topo = np.ones((height*2, height, 3), dtype=np.uint8) * 255
        map_img_p_intra_lane = np.ones((height*2, height, 3), dtype=np.uint8) * 255
        map_img_p_mixed = np.ones((height*2, height, 3), dtype=np.uint8) * 255
        
        # GT的4张图
        map_img_gt_no_topo = np.ones((height*2, height, 3), dtype=np.uint8) * 255
        map_img_gt = np.ones((height*2, height, 3), dtype=np.uint8) * 255
        map_img_gt_intra_lane = np.ones((height*2, height, 3), dtype=np.uint8) * 255
        map_img_gt_mixed = np.ones((height*2, height, 3), dtype=np.uint8) * 255

        # ========== 处理GT ==========
        gt_total_edges = 0
        gt_intra_lane_edges = []
        gt_cross_lane_edges = []
        
        if gt is not None:
            gt_total_edges = gt_graph.number_of_edges()
            
            # 1. GT无拓扑中心线（从gt点集绘制）
            for idx, gt_item in enumerate(gt):
                gt_pts = gt_item['pts']  # shape: (num_points, 2)
                color = LINE_COLORS[idx % len(LINE_COLORS)]
                for j in range(len(gt_pts) - 1):
                    x1 = gt_pts[j, 0] * scale + height // 2
                    y1 = height - gt_pts[j, 1] * scale
                    x2 = gt_pts[j+1, 0] * scale + height // 2
                    y2 = height - gt_pts[j+1, 1] * scale
                    cv2.line(map_img_gt_no_topo, 
                            (np.int32(x1), np.int32(y1)), 
                            (np.int32(x2), np.int32(y2)), 
                            color, 
                            thickness=2)
            
            # 2. 分类GT图的边
            for e in gt_graph.edges():
                node_diff = abs(e[1] - e[0])
                if node_diff == 1:
                    gt_intra_lane_edges.append(e)
                else:
                    gt_cross_lane_edges.append(e)
            
            # 3. GT完整拓扑图
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
                        thickness=3,
                        tipLength=float(20/(distance+1)))
            
            # 4. GT车道内连续边（绿色）
            for e in gt_intra_lane_edges:
                x1 = gt_graph.nodes[e[0]]['pos'][0] *scale
                y1 = gt_graph.nodes[e[0]]['pos'][1] *scale
                x2 = gt_graph.nodes[e[1]]['pos'][0] *scale
                y2 = gt_graph.nodes[e[1]]['pos'][1] *scale
                x1 += height//2
                y1 = height - y1
                x2 += height//2
                y2 = height - y2
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                cv2.circle(map_img_gt_intra_lane, (np.int32(x1), np.int32(y1)), 8, (0, 255, 0), -1)
                cv2.circle(map_img_gt_intra_lane, (np.int32(x2), np.int32(y2)), 8, (0, 255, 0), -1)
                cv2.arrowedLine(map_img_gt_intra_lane, 
                        (np.int32(x1), np.int32(y1)), 
                        (np.int32(x2), np.int32(y2)), 
                        color = (0, 128, 0),
                        thickness=3,
                        tipLength=float(20/(distance+1)))
                
                # 同时绘制到mixed图
                cv2.circle(map_img_gt_mixed, (np.int32(x1), np.int32(y1)), 8, (0, 255, 0), -1)
                cv2.circle(map_img_gt_mixed, (np.int32(x2), np.int32(y2)), 8, (0, 255, 0), -1)
                cv2.arrowedLine(map_img_gt_mixed, 
                        (np.int32(x1), np.int32(y1)), 
                        (np.int32(x2), np.int32(y2)), 
                        color = (0, 128, 0),
                        thickness=3,
                        tipLength=float(20/(distance+1)))
            
            # 5. GT跨车道边（橙色）到mixed图
            for e in gt_cross_lane_edges:
                x1 = gt_graph.nodes[e[0]]['pos'][0] *scale
                y1 = gt_graph.nodes[e[0]]['pos'][1] *scale
                x2 = gt_graph.nodes[e[1]]['pos'][0] *scale
                y2 = gt_graph.nodes[e[1]]['pos'][1] *scale
                x1 += height//2
                y1 = height - y1
                x2 += height//2
                y2 = height - y2
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                cv2.circle(map_img_gt_mixed, (np.int32(x1), np.int32(y1)), 8, (255, 165, 0), -1)
                cv2.circle(map_img_gt_mixed, (np.int32(x2), np.int32(y2)), 8, (255, 165, 0), -1)
                cv2.arrowedLine(map_img_gt_mixed, 
                        (np.int32(x1), np.int32(y1)), 
                        (np.int32(x2), np.int32(y2)), 
                        color = (255, 140, 0),
                        thickness=3,
                        tipLength=float(20/(distance+1)))
        
        # ========== 处理预测结果 ==========
        total_edges = 0
        intra_lane_edges = []
        cross_lane_edges = []
        
        # 1. 预测无拓扑中心线
        for idx, pred in enumerate(line_preds):
            line_pts = pred['line']  # shape: (num_points, 2)
            color = LINE_COLORS[idx % len(LINE_COLORS)]
            for j in range(len(line_pts) - 1):
                x1 = line_pts[j, 0] * scale + height // 2
                y1 = height - line_pts[j, 1] * scale
                x2 = line_pts[j+1, 0] * scale + height // 2
                y2 = height - line_pts[j+1, 1] * scale
                cv2.line(map_img_p_no_topo, 
                        (np.int32(x1), np.int32(y1)), 
                        (np.int32(x2), np.int32(y2)), 
                        color, 
                        thickness=2)
        
        # 2. 分类预测图的边
        total_edges = graph.number_of_edges()
        for e in graph.edges():
            node_diff = abs(e[1] - e[0])
            if node_diff == 1:
                intra_lane_edges.append(e)
            else:
                cross_lane_edges.append(e)
        
        # 3. 预测完整拓扑图
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
                    thickness=3,
                    tipLength=float(20/(distance+1)))
        
        # 4. 预测车道内连续边（绿色）
        for e in intra_lane_edges:
            x1 = graph.nodes[e[0]]['pos'][0] *scale
            y1 = graph.nodes[e[0]]['pos'][1] *scale
            x2 = graph.nodes[e[1]]['pos'][0] *scale
            y2 = graph.nodes[e[1]]['pos'][1] *scale
            x1 += height//2
            y1 = height - y1
            x2 += height//2
            y2 = height - y2
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            cv2.circle(map_img_p_intra_lane, (np.int32(x1), np.int32(y1)), 8, (0, 255, 0), -1)
            cv2.circle(map_img_p_intra_lane, (np.int32(x2), np.int32(y2)), 8, (0, 255, 0), -1)
            cv2.arrowedLine(map_img_p_intra_lane, 
                    (np.int32(x1), np.int32(y1)), 
                    (np.int32(x2), np.int32(y2)), 
                    color = (0, 128, 0),
                    thickness=3,
                    tipLength=float(20/(distance+1)))
            
            # 同时绘制到mixed图
            cv2.circle(map_img_p_mixed, (np.int32(x1), np.int32(y1)), 8, (0, 255, 0), -1)
            cv2.circle(map_img_p_mixed, (np.int32(x2), np.int32(y2)), 8, (0, 255, 0), -1)
            cv2.arrowedLine(map_img_p_mixed, 
                    (np.int32(x1), np.int32(y1)), 
                    (np.int32(x2), np.int32(y2)), 
                    color = (0, 128, 0),
                    thickness=3,
                    tipLength=float(20/(distance+1)))
        
        # 5. 预测跨车道边（橙色）到mixed图
        for e in cross_lane_edges:
            x1 = graph.nodes[e[0]]['pos'][0] *scale
            y1 = graph.nodes[e[0]]['pos'][1] *scale
            x2 = graph.nodes[e[1]]['pos'][0] *scale
            y2 = graph.nodes[e[1]]['pos'][1] *scale
            x1 += height//2
            y1 = height - y1
            x2 += height//2
            y2 = height - y2
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            cv2.circle(map_img_p_mixed, (np.int32(x1), np.int32(y1)), 8, (255, 165, 0), -1)
            cv2.circle(map_img_p_mixed, (np.int32(x2), np.int32(y2)), 8, (255, 165, 0), -1)
            cv2.arrowedLine(map_img_p_mixed, 
                    (np.int32(x1), np.int32(y1)), 
                    (np.int32(x2), np.int32(y2)), 
                    color = (255, 140, 0),
                    thickness=3,
                    tipLength=float(20/(distance+1)))
        
        print(f"\n[Edge Statistics]")
        print(f"  GT    - Total: {gt_total_edges}, Intra: {len(gt_intra_lane_edges)}, Cross: {len(gt_cross_lane_edges)}")
        print(f"  Pred  - Total: {total_edges}, Intra: {len(intra_lane_edges)}, Cross: {len(cross_lane_edges)}")


        f, fr, fl, b, bl, br = imgs
        canvas = cv2.vconcat([cv2.hconcat([fl, f, fr]), cv2.hconcat([bl, b, br])])
        
        # 使用 matplotlib 显示对比图（2行4列布局：第1行GT，第2行预测）
        fig, axes = plt.subplots(2, 4, figsize=(32, 12))
        
        # ========== 第1行：Ground Truth ==========
        axes[0, 0].imshow(cv2.cvtColor(map_img_gt_no_topo, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('GT: No Topology', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(map_img_gt, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f'GT: Full Topology\n({gt_total_edges} edges)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(cv2.cvtColor(map_img_gt_intra_lane, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title(f'GT: Intra-Lane Only (Green)\n({len(gt_intra_lane_edges)} edges)', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(cv2.cvtColor(map_img_gt_mixed, cv2.COLOR_BGR2RGB))
        axes[0, 3].set_title(f'GT: Mixed: Intra(Green) + Cross(Orange)\n({len(gt_intra_lane_edges)}+{len(gt_cross_lane_edges)} edges)', fontsize=14, fontweight='bold')
        axes[0, 3].axis('off')
        
        # ========== 第2行：Predictions ==========
        axes[1, 0].imshow(cv2.cvtColor(map_img_p_no_topo, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Pred: No Topology', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cv2.cvtColor(map_img_p, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Pred: Full Topology\n({total_edges} edges)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(cv2.cvtColor(map_img_p_intra_lane, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f'Pred: Intra-Lane Only (Green)\n({len(intra_lane_edges)} edges)', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(cv2.cvtColor(map_img_p_mixed, cv2.COLOR_BGR2RGB))
        axes[1, 3].set_title(f'Pred: Mixed: Intra(Green) + Cross(Orange)\n({len(intra_lane_edges)}+{len(cross_lane_edges)} edges)', fontsize=14, fontweight='bold')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 仍然保存原有的拼接图像
        canvas_full = cv2.hconcat([canvas, map_img_gt.astype(np.float32), map_img_p.astype(np.float32)])
        cv2.imwrite(out_file, canvas_full)

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
