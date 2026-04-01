from .transformer import JAPerceptionTransformer

# MapTR baseline transformer（不含kp_bev_preds）
from .MapTR_transformer import MapTRPerceptionTransformer

from .decoder import MapTRDecoder
from .decoder_dam import MapTRDecoderWithDAM  # 支持DAM分析的解码器
from .builder import build_fuser
from .encoder import LSSTransform
from .geometry_kernel_attention import GeometrySptialCrossAttention, \
                                        GeometryKernelAttention

# Sparse DETR相关模块 - 全部从decoder_sparse导入
from .decoder_sparse import (
    MapTRDecoderSparse, 
    DetrTransformerDecoderLayerSparse,
    CustomMSDeformableAttentionSparse,
    attn_map_to_flat_grid,
    idx_to_flat_grid,
    compute_corr,
    aggregate_cross_attn_weights
)
from .mask_predictor_sparse import MaskPredictor, build_mask_predictor
from .encoder_sparse import LSSTransformSparse, gather_tokens_by_indices, scatter_tokens_by_indices
from .transformer_sparse import JAPerceptionTransformerSparse
from .bevformer_encoder_sparse import BEVFormerEncoderSparse, BEVFormerLayerSparse

