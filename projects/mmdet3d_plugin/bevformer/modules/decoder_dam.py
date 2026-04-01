"""
支持DAM分析的CustomMSDeformableAttention模块

这个文件扩展了CustomMSDeformableAttention，添加了保存注意力权重的功能，
用于DAM (Decoder cross-Attention Map) 分析。
"""

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
import math
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule
from mmcv.utils import deprecated_api_warning

from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32


@ATTENTION.register_module()
class CustomMSDeformableAttentionWithDAM(BaseModule):
    """
    支持DAM分析的可变形注意力模块
    
    在原有CustomMSDeformableAttention基础上增加：
    - 保存注意力权重用于DAM分析
    - 提供获取注意力权重的接口
    
    使用方式:
    1. 在推理时启用save_attn_weights模式
    2. 前向传播后通过get_last_attn_weights()获取注意力权重
    3. 用于计算编码器token的引用统计
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None,
                 save_attn_weights=False):  # 新增参数
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # DAM分析相关
        self.save_attn_weights = save_attn_weights
        self.last_attn_weights = None  # 保存最后一次的注意力权重

        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """初始化模块参数"""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',
                **kwargs):
        """
        多尺度可变形注意力的前向传播
        
        Args:
            query (Tensor): 查询张量，形状 (num_query, bs, embed_dims)
            key (Tensor): 键张量
            value (Tensor): 值张量，形状 (num_key, bs, embed_dims)
            identity (Tensor): 用于残差连接的张量
            query_pos (Tensor): 查询的位置编码
            key_padding_mask (Tensor): 键的padding mask
            reference_points (Tensor): 归一化的参考点，形状 (bs, num_query, num_levels, 2)
            spatial_shapes (Tensor): 不同层级特征的空间形状
            level_start_index (Tensor): 每个层级的起始索引
            flag (str): 'encoder' 或 'decoder'
            
        Returns:
            Tensor: 前向传播结果，形状 [num_query, bs, embed_dims]
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # 转换为 (bs, num_query, embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        # 投影value
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        # 计算采样偏移和注意力权重
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        
        # Softmax归一化注意力权重
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        
        # 计算采样位置
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        
        # 如果启用DAM分析，保存采样位置和注意力权重用于分析
        if self.save_attn_weights:
            # 保存以下信息用于DAM分析：
            # 1. sampling_locations: 采样位置 [bs, num_query, num_heads, num_levels, num_points, 2]
            # 2. attention_weights: 注意力权重 [bs, num_query, num_heads, num_levels, num_points]
            # 3. spatial_shapes: 特征图尺寸 [num_levels, 2]
            self.last_attn_weights = {
                'sampling_locations': sampling_locations.detach().clone(),
                'attention_weights': attention_weights.detach().clone(),
                'spatial_shapes': spatial_shapes.clone(),
            }
        
        # 执行多尺度可变形注意力
        if torch.cuda.is_available() and value.is_cuda:
            # 使用FP32版本以保持稳定性
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        # 输出投影
        output = self.output_proj(output)

        if not self.batch_first:
            # 转换回 (num_query, bs, embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

    def get_last_attn_weights(self):
        """
        获取最后一次前向传播的注意力权重
        
        Returns:
            Tensor or None: 注意力权重，形状 [bs, num_query, num_heads, num_levels * num_points]
        """
        return self.last_attn_weights
    
    def enable_save_attn_weights(self):
        """启用保存注意力权重"""
        self.save_attn_weights = True
        print("Attention weights saving enabled for DAM analysis")
    
    def disable_save_attn_weights(self):
        """禁用保存注意力权重"""
        self.save_attn_weights = False
        self.last_attn_weights = None
        print("Attention weights saving disabled")
