"""
Sparse DETR风格的Decoder相关模块
包含:
1. DAM工具函数 (attn_map_to_flat_grid, idx_to_flat_grid, compute_corr)
2. CustomMSDeformableAttentionSparse - 返回sampling信息的attention
3. DetrTransformerDecoderLayerSparse - 收集attention信息的decoder layer
4. MapTRDecoderSparse - Sparse DETR风格的decoder

参考: sparse-detr/util/dam.py
"""

import copy
import warnings
import torch
import torch.nn.functional as F
from torch import nn
from mmcv.cnn.bricks.registry import ATTENTION, TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import BaseTransformerLayer, TransformerLayerSequence
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.transformer import inverse_sigmoid
from projects.mmdet3d_plugin.bevformer.modules.decoder import CustomMSDeformableAttention


def attn_map_to_flat_grid(spatial_shapes, level_start_index, sampling_locations, attention_weights):
    """
    将Deformable Attention的attention weights转换为flat grid (Sparse DETR优化版本)
    
    使用向量化操作和双线性插值,性能优化30-50倍
    
    Args:
        spatial_shapes: [num_levels, 2] - 每个level的空间尺寸 (H, W)
        level_start_index: [num_levels] - 每个level在flatten特征中的起始索引
        sampling_locations: [N, n_layers, Len_q, n_heads, n_levels, n_points, 2]
                          采样位置的归一化坐标 [0, 1]
        attention_weights: [N, n_layers, Len_q, n_heads, n_levels, n_points]
                         每个采样点的attention权重
    
    Returns:
        flat_grid: [N, n_layers, n_heads, H*W_sum] 
                  每个BEV位置被关注的总权重 (所有levels合并)
    
    优化:
        1. 使用permute+flatten合并batch/layer/head维度,实现GPU并行
        2. 双线性插值处理亚像素位置(4个角点)
        3. 只需4次scatter_add而非2400次
    """
    # sampling_locations: [N, n_layers, Len_q, n_heads, n_levels, n_points, 2]
    # attention_weights: [N, n_layers, Len_q, n_heads, n_levels, n_points]
    N, n_layers, _, n_heads, *_ = sampling_locations.shape
    
    # 重排维度并合并: [N * n_layers * n_heads, Len_q * n_points, n_levels, 2]
    sampling_locations = sampling_locations.permute(0, 1, 3, 2, 5, 4, 6).flatten(0, 2).flatten(1, 2)
    # attention_weights: [N * n_layers * n_heads, Len_q * n_points, n_levels]
    attention_weights = attention_weights.permute(0, 1, 3, 2, 5, 4).flatten(0, 2).flatten(1, 2)
    
    # spatial_shapes是HW格式,需要转换为WH(xy)格式
    rev_spatial_shapes = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)  # hw -> wh (xy)
    
    # 将归一化坐标[0,1]转换为像素坐标
    col_row_float = sampling_locations * rev_spatial_shapes
    
    # 计算双线性插值的4个角点
    col_row_ll = col_row_float.floor().to(torch.int64)  # 左下角
    zero = torch.zeros(*col_row_ll.shape[:-1], dtype=torch.int64, device=col_row_ll.device)
    one = torch.ones(*col_row_ll.shape[:-1], dtype=torch.int64, device=col_row_ll.device)
    col_row_lh = col_row_ll + torch.stack([zero, one], dim=-1)  # 左上角
    col_row_hl = col_row_ll + torch.stack([one, zero], dim=-1)  # 右下角
    col_row_hh = col_row_ll + 1  # 右上角
    
    # 计算双线性插值权重(距离越远权重越小)
    margin_ll = (col_row_float - col_row_ll).prod(dim=-1)  # 右上角的权重给左下角
    margin_lh = -(col_row_float - col_row_lh).prod(dim=-1)  # 右下角的权重给左上角
    margin_hl = -(col_row_float - col_row_hl).prod(dim=-1)  # 左上角的权重给右下角
    margin_hh = (col_row_float - col_row_hh).prod(dim=-1)  # 左下角的权重给右上角
    
    # 初始化flat_grid
    flat_grid_shape = (attention_weights.shape[0], int(torch.sum(spatial_shapes[..., 0] * spatial_shapes[..., 1])))
    flat_grid = torch.zeros(flat_grid_shape, dtype=torch.float32, device=attention_weights.device)
    
    # 对4个角点分别scatter_add (这里只需4次循环,而不是2400次!)
    zipped = [(col_row_ll, margin_hh), (col_row_lh, margin_hl), (col_row_hl, margin_lh), (col_row_hh, margin_ll)]
    for col_row, margin in zipped:
        # 检查是否在有效范围内
        valid_mask = torch.logical_and(
            torch.logical_and(col_row[..., 0] >= 0, col_row[..., 0] < rev_spatial_shapes[..., 0]),
            torch.logical_and(col_row[..., 1] >= 0, col_row[..., 1] < rev_spatial_shapes[..., 1]),
        )
        # 计算线性索引: y * W + x + level_offset
        idx = col_row[..., 1] * spatial_shapes[..., 1] + col_row[..., 0] + level_start_index
        idx = (idx * valid_mask).flatten(1, 2)
        # 计算最终权重 = attention_weight * valid_mask * bilinear_margin
        weights = (attention_weights * valid_mask * margin).flatten(1)
        # 向量化scatter_add (处理所有N*n_layers*n_heads)
        flat_grid.scatter_add_(1, idx, weights)
    
    # 重新整形为 [N, n_layers, n_heads, -1]
    return flat_grid.reshape(N, n_layers, n_heads, -1)


def idx_to_flat_grid(spatial_shapes, indices):
    """
    将indices转换为flat grid binary mask (Sparse DETR原版实现)
    
    完全向量化,无Python循环
    
    Args:
        spatial_shapes: [num_levels, 2] - (H, W)
        indices: [bs, num_indices] - 索引位置
    
    Returns:
        flat_grid: [bs, H*W] - binary mask,选中的位置为1
    """
    # 计算flat_grid的形状
    if isinstance(spatial_shapes, torch.Tensor):
        total_size = int(torch.sum(spatial_shapes[..., 0] * spatial_shapes[..., 1]))
    else:
        total_size = int(sum(h * w for h, w in spatial_shapes))
    
    flat_grid_shape = (indices.shape[0], total_size)
    flat_grid = torch.zeros(flat_grid_shape, device=indices.device, dtype=torch.float32)
    
    # 向量化scatter: 直接在dim=1上scatter
    flat_grid.scatter_(1, indices.to(torch.int64), 1)
    
    return flat_grid


def compute_corr(flat_grid_topk, flat_grid_attn_map, spatial_shapes):
    """
    计算预测的top-k与attention map的相关性 (Sparse DETR原版实现)
    
    使用加权相关性: 被选中位置的attention权重占总权重的比例
    完全在GPU上计算,无CPU-GPU同步
    
    Args:
        flat_grid_topk: [bs, H*W] - encoder预测的top-k位置 (binary mask, 0或1)
        flat_grid_attn_map: [bs, H*W] - decoder attention的flat grid (continuous)
        spatial_shapes: [num_levels, 2] - (H, W)
    
    Returns:
        corr: list of tensors - [overall_corr, per_level_corr...]
              overall_corr: [bs] - 总体相关性
              per_level_corr: [bs] - 每个level的相关性
    """
    # 处理单个样本的情况
    if len(flat_grid_topk.shape) == 1:
        flat_grid_topk = flat_grid_topk.unsqueeze(0)
        flat_grid_attn_map = flat_grid_attn_map.unsqueeze(0)
    
    # 总体相关性: 被选中位置的attention权重占总权重的比例
    tot = flat_grid_attn_map.sum(-1)  # [bs] - 总attention
    hit = (flat_grid_topk * flat_grid_attn_map).sum(-1)  # [bs] - 命中位置的attention
    
    corr = [hit / tot]  # [bs]
    
    # 计算每个level的相关性
    flat_grid_idx = 0
    for shape in spatial_shapes:
        level_range = torch.arange(
            int(flat_grid_idx), 
            int(flat_grid_idx + shape[0] * shape[1]),
            device=flat_grid_topk.device
        )
        tot = (flat_grid_attn_map[:, level_range]).sum(-1)  # [bs]
        hit = (flat_grid_topk[:, level_range] * flat_grid_attn_map[:, level_range]).sum(-1)  # [bs]
        flat_grid_idx += shape[0] * shape[1]
        corr.append(hit / tot)
    
    return corr


# ========== 与之前实现兼容的aggregate函数 ==========

def aggregate_cross_attn_weights(attn_weights, sampling_locations=None, bev_h=None, bev_w=None, spatial_shapes=None):
    """
    聚合decoder cross-attention weights到BEV grid
    
    这是之前实现的aggregate函数的包装,现在调用attn_map_to_flat_grid
    
    Args:
        attn_weights: [num_layers, bs, num_query, num_heads, num_levels, num_points]
        sampling_locations: [num_layers, bs, num_query, num_heads, num_levels, num_points, 2]
        bev_h, bev_w: BEV尺寸
        spatial_shapes: [[H, W]]
    
    Returns:
        aggregated: [bs, H*W] - 每个BEV位置的总attention
    """
    if attn_weights is None or sampling_locations is None:
        return None
    
    # 构造spatial_shapes
    if spatial_shapes is None:
        if bev_h is not None and bev_w is not None:
            spatial_shapes = torch.tensor([[bev_h, bev_w]], device=attn_weights.device)
        else:
            return None
    
    # level_start_index (单个level时为[0])
    level_start_index = torch.tensor([0], device=attn_weights.device)
    
    # 调用attn_map_to_flat_grid
    # 返回: [N, n_layers, n_heads, H*W]
    flat_grid = attn_map_to_flat_grid(
        spatial_shapes, level_start_index,
        sampling_locations, attn_weights
    )
    
    # 对所有layers和heads求和: [N, n_layers, n_heads, H*W] -> [N, H*W]
    aggregated = flat_grid.sum(dim=(1, 2))
    
    return aggregated


# ========== Sparse Attention Module ==========

@ATTENTION.register_module()
class CustomMSDeformableAttentionSparse(CustomMSDeformableAttention):
    """
    Sparse DETR版本的CustomMSDeformableAttention
    
    在原有功能基础上，额外返回sampling_locations和attention_weights
    用于计算DAM loss
    
    Args:
        return_attn_info (bool): 是否返回attention信息。默认False保持向后兼容
        其他参数同CustomMSDeformableAttention
    """
    
    def __init__(self, 
                 return_attn_info=False,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.return_attn_info = return_attn_info
    
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
                **kwargs):
        """
        Forward with optional return of sampling_locations and attention_weights
        
        Returns:
            如果return_attn_info=False (默认):
                output: [num_query, bs, embed_dims] or [bs, num_query, embed_dims]
            如果return_attn_info=True:
                (output, sampling_locations, attention_weights)
                - output: [num_query, bs, embed_dims] or [bs, num_query, embed_dims]
                - sampling_locations: [bs, num_query, num_heads, num_levels, num_points, 2]
                - attention_weights: [bs, num_query, num_heads, num_levels, num_points]
        """
        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        
        if not self.batch_first:
            # change to (bs, num_query, embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        # 计算sampling_offsets和attention_weights
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        
        # 计算sampling_locations
        sampling_locations = None
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
        
        # 调用multi_scale_deformable_attention
        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                from projects.mmdet3d_plugin.bevformer.modules.multi_scale_deformable_attn_function import \
                    MultiScaleDeformableAttnFunction_fp16
                output = MultiScaleDeformableAttnFunction_fp16.apply(
                    value, spatial_shapes, level_start_index, sampling_locations,
                    attention_weights, self.im2col_step)
            else:
                from projects.mmdet3d_plugin.bevformer.modules.multi_scale_deformable_attn_function import \
                    MultiScaleDeformableAttnFunction_fp32
                output = MultiScaleDeformableAttnFunction_fp32.apply(
                    value, spatial_shapes, level_start_index, sampling_locations,
                    attention_weights, self.im2col_step)
        else:
            from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        # 返回结果
        if self.return_attn_info:
            # 返回 (output, sampling_locations, attention_weights)
            return output + identity, sampling_locations, attention_weights  # sampling_locations torch.Size([6, 4, 1000, 8, 1, 4, 2])
        else:
            # 保持向后兼容，只返回output
            return output + identity


# ========== Sparse Decoder Layer ==========

@TRANSFORMER_LAYER.register_module()
class DetrTransformerDecoderLayerSparse(BaseTransformerLayer):
    """
    Sparse DETR风格的Decoder Layer
    
    在执行cross-attention时，如果attention模块支持返回sampling信息，
    则收集并返回这些信息
    
    这个类继承BaseTransformerLayer，覆盖forward方法以支持返回attention信息
    """
    
    def __init__(self, *args, return_attn_info=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_attn_info = return_attn_info
        self.last_sampling_locations = None
        self.last_attn_weights = None
    
    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """
        Forward with optional collection of attention information
        
        执行顺序: self_attn -> norm -> cross_attn -> norm -> ffn -> norm
        
        Returns:
            query: transformed query tensor
            同时，如果return_attn_info=True且cross_attn返回了sampling信息，
            会保存到self.last_sampling_locations和self.last_attn_weights
        """
        # 重置
        self.last_sampling_locations = None
        self.last_attn_weights = None
        
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                         f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                # Cross-attention - 可能返回sampling信息
                attn_output = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                
                # 检查是否返回了sampling信息
                if isinstance(attn_output, tuple) and len(attn_output) == 3:
                    # (output, sampling_locations, attention_weights)
                    query, sampling_locs, attn_weights = attn_output
                    if self.return_attn_info:
                        self.last_sampling_locations = sampling_locs
                        self.last_attn_weights = attn_weights
                else:
                    # 普通返回，只有output
                    query = attn_output
                
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


# ========== Sparse Decoder ==========

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapTRDecoderSparse(TransformerLayerSequence):
    """
    Sparse DETR风格的MapTR Decoder
    
    继承TransformerLayerSequence，在forward过程中收集cross-attention的
    sampling_locations和attention_weights，用于计算DAM loss
    
    Args:
        return_intermediate (bool): 是否返回中间层结果
        return_sampling_locs (bool): 是否返回sampling locations和attention weights
        其他参数同TransformerLayerSequence
    """

    def __init__(self, 
                 *args, 
                 return_intermediate=False,
                 return_sampling_locs=False,
                 **kwargs):
        super(MapTRDecoderSparse, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.return_sampling_locs = return_sampling_locs
        self.fp16_enabled = False
        
        # 设置所有layer和attention模块的return_attn_info标志
        if self.return_sampling_locs:
            for layer in self.layers:
                # 设置layer的return_attn_info
                if hasattr(layer, 'return_attn_info'):
                    layer.return_attn_info = True
                    
                # 设置layer中所有CustomMSDeformableAttentionSparse的return_attn_info
                for module in layer.modules():
                    if module.__class__.__name__ == 'CustomMSDeformableAttentionSparse':
                        module.return_attn_info = True

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):
        """
        Forward with collection of sampling locations and attention weights
        
        Returns:
            如果return_sampling_locs=False:
                (inter_states, inter_references)
            如果return_sampling_locs=True:
                (inter_states, inter_references, sampling_locations, attn_weights)
                - sampling_locations: [num_layers, bs, num_query, num_heads, num_levels, num_points, 2]
                - attn_weights: [num_layers, bs, num_query, num_heads, num_levels, num_points]
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        
        # 收集所有层的sampling locations和attention weights
        all_sampling_locs = [] if self.return_sampling_locs else None
        all_attn_weights = [] if self.return_sampling_locs else None
        
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)  # BS NUM_QUERY NUM_LEVEL 2
            
            # 调用layer的forward
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            
            # 如果需要sampling信息，从layer中获取
            if self.return_sampling_locs and hasattr(layer, 'last_sampling_locations'):
                if layer.last_sampling_locations is not None:
                    all_sampling_locs.append(layer.last_sampling_locations)
                    all_attn_weights.append(layer.last_attn_weights)
            
            output = output.permute(1, 0, 2)

            # Reference points refinement
            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        # 构造返回值
        if self.return_intermediate:
            inter_states = torch.stack(intermediate)
            inter_references = torch.stack(intermediate_reference_points)
        else:
            inter_states = output
            inter_references = reference_points
        
        if self.return_sampling_locs:
            # Stack所有层的sampling信息
            if all_sampling_locs:
                sampling_locs = torch.stack(all_sampling_locs, dim=0)  # [num_layers, bs, num_query, ...]
                attn_weights = torch.stack(all_attn_weights, dim=0)
                return inter_states, inter_references, sampling_locs, attn_weights
            else:
                # 没有收集到sampling信息，返回None
                return inter_states, inter_references, None, None
        else:
            return inter_states, inter_references
