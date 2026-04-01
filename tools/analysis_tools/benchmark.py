# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys
import time
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

# Ensure this benchmark script prefers the local CGNet/MMDetection3D codebase.
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CUR_DIR, '..', '..'))
_MMDET3D_ROOT = os.path.join(_PROJECT_ROOT, 'mmdetection3d')
for _p in (_PROJECT_ROOT, _MMDET3D_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from tools.misc.fuse_conv_bn import fuse_module


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--samples', type=int, default=2000, help='samples to benchmark')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plugin dir so custom registry entries are available.
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir).split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            importlib.import_module(_module_path)
        else:
            _module_dir = os.path.dirname(args.config).split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)

    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0

    # benchmark with several samples and take the average
    for i, data in enumerate(data_loader):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done image [{i + 1:<3}/ {args.samples}], '
                      f'fps: {fps:.1f} img / s')

        if (i + 1) == args.samples:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.1f} img / s')
            break


if __name__ == '__main__':
    main()
