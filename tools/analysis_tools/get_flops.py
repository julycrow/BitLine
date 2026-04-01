# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import importlib
import os
import sys

import numpy as np
import torch
from mmcv import Config, DictAction

# Ensure this script prefers the local CGNet/MMDetection3D codebase.
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CUR_DIR, '..', '..'))
_MMDET3D_ROOT = os.path.join(_PROJECT_ROOT, 'mmdetection3d')
for _p in (_PROJECT_ROOT, _MMDET3D_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mmdet3d.models import build_model

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[40000, 4],
        help='input point cloud size')
    parser.add_argument(
        '--modality',
        type=str,
        default='point',
        choices=['point', 'image', 'multi'],
        help='input data modality')
    parser.add_argument(
        '--num-cams',
        type=int,
        default=6,
        help='number of cameras for image/multi modality')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='device used to build dummy input')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import plugin modules to update registry for custom models/datasets.
    if hasattr(cfg, 'plugin') and cfg.plugin:
        if hasattr(cfg, 'plugin_dir'):
            module_path = os.path.dirname(cfg.plugin_dir).replace('/', '.')
            importlib.import_module(module_path)
        else:
            module_path = os.path.dirname(args.config).replace('/', '.')
            importlib.import_module(module_path)

    # Auto switch for camera-based CGNet/BEVFormer style configs.
    model_type = str(cfg.model.get('type', ''))
    if args.modality == 'point' and model_type in ('CGNet', 'BEVFormer'):
        print('Detected camera-based model. Auto switch modality: point -> image')
        args.modality = 'image'
        if args.shape == [40000, 4]:
            # A practical default for nuScenes-like camera inputs.
            args.shape = [900, 1600]

    if args.modality == 'point':
        assert len(args.shape) == 2, 'invalid input shape'
        input_shape = tuple(args.shape)
    elif args.modality == 'image':
        if len(args.shape) == 1:
            h, w = args.shape[0], args.shape[0]
        elif len(args.shape) == 2:
            h, w = args.shape
        else:
            raise ValueError('invalid input shape')
        # forward_dummy for CGNet expects img with shape [B, num_cams, 3, H, W]
        input_shape = (args.num_cams, 3, h, w)
    elif args.modality == 'multi':
        if len(args.shape) == 1:
            h, w = args.shape[0], args.shape[0]
        elif len(args.shape) == 2:
            h, w = args.shape
        else:
            raise ValueError('invalid input shape')
        # For CGNet-like models, use image branch dummy input for FLOPs.
        input_shape = (args.num_cams, 3, h, w)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if args.device == 'cuda' and torch.cuda.is_available():
        model.cuda()
    model.eval()

    input_constructor = None
    if not hasattr(model, 'forward_dummy'):
        raise NotImplementedError(
            'FLOPs counter is currently not supported for {}'.format(
                model.__class__.__name__))

    if model_type in ('CGNet', 'BEVFormer'):
        def _forward_dummy_with_meta(img):
            # img: [B, num_cams, 3, H, W]
            bsz, num_cams, _, img_h, img_w = img.shape
            eye = np.eye(4, dtype=np.float32)

            batch_metas = []
            for b in range(bsz):
                batch_metas.append(
                    dict(
                        scene_token=f'flops_scene_{b}',
                        can_bus=np.zeros(18, dtype=np.float32),
                        img_shape=[(img_h, img_w, 3) for _ in range(num_cams)],
                        lidar2img=[eye.copy() for _ in range(num_cams)],
                        camera2ego=[eye.copy() for _ in range(num_cams)],
                        camera_intrinsics=[eye.copy() for _ in range(num_cams)],
                        img_aug_matrix=[eye.copy() for _ in range(num_cams)],
                        lidar2ego=eye.copy(),
                    ))
            # Keep the same nesting as test-time forward:
            # - img_metas: [num_augs][batch]
            # - img: [num_augs] where each item is a tensor [B, num_cams, 3, H, W]
            with torch.no_grad():
                return model.forward_test(
                    img=[img], img_metas=[batch_metas], points=[None])

        model.forward = _forward_dummy_with_meta
    else:
        model.forward = model.forward_dummy

    if args.modality in ('image', 'multi'):
        def input_constructor(input_res):
            device = next(model.parameters()).device
            img = torch.randn((1, ) + tuple(input_res), device=device)
            return dict(img=img)

    flops, params = get_model_complexity_info(
        model, input_shape, input_constructor=input_constructor)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
