# Copyright (C) 2024 Xiaomi Corporation.
# Sparse DETR Implementation - Sparse BEVFormer Encoder
# 基于BEVFormerEncoder实现Sparse DETR稀疏化

import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE, TRANSFORMER_LAYER
from mmcv.runner import auto_fp16
from projects.mmdet3d_plugin.bevformer.modules.encoder import BEVFormerEncoder, BEVFormerLayer


def gather_tokens_by_indices(tokens, topk_indices):
    """
    根据top-k索引gather tokens
    
    Args:
        tokens: [B, L, C] 完整的token序列
        topk_indices: [B, K] top-k索引
        
    Returns:
        gathered_tokens: [B, K, C] 选中的tokens
    """
    B, L, C = tokens.shape  # 4,20000,8
    K = topk_indices.shape[1]  # 10000
    
    # 扩展索引维度以匹配token维度
    indices_expanded = topk_indices.unsqueeze(-1).expand(B, K, C)  # torch.Size([4, 10000, 8])
    
    # 使用gather提取对应位置的tokens
    gathered = torch.gather(tokens, dim=1, index=indices_expanded)  # test torch.Size([1, 10000, 8])
    
    return gathered


def scatter_tokens_by_indices(full_tokens, sparse_tokens, topk_indices):
    """
    将更新后的稀疏tokens scatter回完整序列
    
    Args:
        full_tokens: [B, L, C] 完整的token序列(作为base)
        sparse_tokens: [B, K, C] 更新后的稀疏tokens
        topk_indices: [B, K] 对应的位置索引
        
    Returns:
        updated_tokens: [B, L, C] 更新后的完整序列
        
    注意: 
        完全避免inplace操作,使用纯函数式方式实现scatter
        参考Sparse DETR的实现方式
    """
    B, L, C = full_tokens.shape
    K = topk_indices.shape[1]
    
    # 扩展索引到所有通道
    indices_expanded = topk_indices.unsqueeze(-1).expand(B, K, C)
    
    # 方法: 创建全0 tensor,scatter更新位置,然后与原tensor组合
    # 1. 创建用于存放sparse_tokens的全0 tensor
    sparse_expanded = torch.zeros_like(full_tokens)
    
    # 2. 将sparse_tokens scatter到对应位置 (这里创建新tensor,不是inplace)
    sparse_expanded = sparse_expanded.scatter(dim=1, index=indices_expanded, src=sparse_tokens)
    
    # 3. 创建mask: 标记哪些位置被更新
    mask = torch.zeros(B, L, dtype=torch.bool, device=full_tokens.device)
    mask = mask.scatter(dim=1, index=topk_indices, src=torch.ones(B, K, dtype=torch.bool, device=full_tokens.device))
    mask = mask.unsqueeze(-1)  # [B, L, 1]
    
    # 4. 使用where组合: 更新位置用sparse_expanded,其他位置用full_tokens
    output = torch.where(mask, sparse_expanded, full_tokens)
    
    return output


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayerSparse(BEVFormerLayer):
    """
    支持稀疏化的BEVFormerLayer
    
    关键改动:
        1. forward接受is_sparse标志
        2. 当is_sparse=True时,从query.shape推断spatial_shapes
        3. 重写temporal self attention部分的spatial_shapes生成
    """
    
    def __init__(self, *args, **kwargs):
        super(BEVFormerLayerSparse, self).__init__(*args, **kwargs)
    
    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                is_sparse=False,  # 新增:标记是否为稀疏模式
                tgt=None,  # 新增:稀疏tokens (cross-attention模式)
                **kwargs):
        """
        稀疏感知的forward
        
        参考 Sparse DETR 的实现:
        - 当 tgt=None: 标准 self-attention (query=key=value)
        - 当 tgt!=None: cross-attention 模式 (tgt 是稀疏 tokens, query 是完整序列)
          * tgt 用于 self-attention 的 query
          * query 用于 cross-attention 的 key/value (src)
        """
        import copy
        import warnings
        
        # 如果提供了tgt,使用cross-attention模式 (Sparse DETR)
        if tgt is not None:
            # tgt: [bs, K, C] 稀疏tokens
            # query: [bs, L, C] 完整序列 (用作 src)
            norm_index = 0
            attn_index = 0
            ffn_index = 0
            identity = tgt
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
                # temporal self attention (tgt做self-attention)
                if layer == 'self_attn':
                    # 稀疏模式: 调整spatial_shapes以匹配实际的value数量
                    # CUDA kernel要求: spatial_shapes.prod() == value的第2维
                    if is_sparse and tgt is not None:
                        K = tgt.shape[1]  # 稀疏tokens数量
                        # 使用[K, 1]让spatial_shapes.prod() = K
                        # 注意: 这会改变offset_normalizer,但对于稀疏tokens是合理的
                        self_spatial_shapes = torch.tensor([[K, 1]], device=tgt.device)
                    else:
                        self_spatial_shapes = torch.tensor([[bev_h, bev_w]], device=tgt.device)
                    
                    tgt = self.attentions[attn_index](
                        tgt,
                        prev_bev,
                        prev_bev,
                        identity if self.pre_norm else None,
                        query_pos=bev_pos,
                        key_pos=bev_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        reference_points=ref_2d,
                        spatial_shapes=self_spatial_shapes,
                        level_start_index=torch.tensor([0], device=tgt.device),
                        **kwargs)
                    attn_index += 1
                    identity = tgt

                elif layer == 'norm':
                    tgt = self.norms[norm_index](tgt)
                    norm_index += 1

                # spatial cross attention (tgt作为query, query作为src)
                elif layer == 'cross_attn':
                    tgt = self.attentions[attn_index](
                        tgt,
                        key,
                        value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=key_pos,
                        reference_points=ref_3d,
                        reference_points_cam=reference_points_cam,
                        mask=mask,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=key_padding_mask,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        **kwargs)
                    attn_index += 1
                    identity = tgt

                elif layer == 'ffn':
                    tgt = self.ffns[ffn_index](
                        tgt, identity if self.pre_norm else None)
                    ffn_index += 1

            return tgt
        
        else:
            # 标准模式:调用父类
            return super().forward(
                query=query,
                key=key,
                value=value,
                bev_pos=bev_pos,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                ref_2d=ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                reference_points_cam=reference_points_cam,
                mask=mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                prev_bev=prev_bev,
                **kwargs
            )


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoderSparse(BEVFormerEncoder):
    """
    稀疏化的BEVFormer Encoder - Sparse DETR实现
    
    核心思路 (对应Sparse DETR的encoder实现):
        1. 在encoder处理前,接收外部传入的top-k索引
        2. 在encoder的每一层中:
           - gather: 提取稀疏tokens
           - 只对稀疏tokens执行attention和FFN
           - scatter: 将更新后的tokens写回完整序列
        3. 最终输出完整的BEV特征(但只有rho%被更新过)
        
    与Sparse DETR对应关系:
        - DeformableTransformerEncoder.forward() 中的稀疏化逻辑
        - 使用gather/scatter机制只更新重要tokens
        - 完整memory保持,确保decoder可访问所有位置
        
    参数:
        继承自BEVFormerEncoder,无需新增参数
        稀疏化通过forward时传入topk_indices控制
    """
    
    def __init__(self, *args, **kwargs):
        super(BEVFormerEncoderSparse, self).__init__(*args, **kwargs)
        # 稀疏化完全由topk_indices参数控制,不需要额外初始化
        # 注意: 配置中应该使用BEVFormerLayerSparse而不是BEVFormerLayer
        
        # 【Encoder辅助头】用于encoder中间层监督
        # 参考Sparse DETR的DeformableTransformerEncoder
        # 辅助头会在forward时通过参数传递,避免参数重复注册
        self.aux_heads = False  # 初始值为False,会在Head的__init__中根据use_enc_aux_loss配置设置为True
        
    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                topk_indices=None,  # [B, K] top-k索引,启用稀疏化
                output_proposals=None,  # [B, H*W, 2] 位置proposals (用于encoder aux loss)
                sparse_token_nums=None,  # [B] 每个样本的稀疏token数量
                cls_branches=None,  # 分类分支 (用于encoder aux loss)
                reg_branches=None,  # 回归分支 (用于encoder aux loss)
                **kwargs):
        """
        稀疏化的前向传播
        
        新增参数:
            topk_indices: [B, K] top-k索引,指示哪些tokens需要更新
            output_proposals: [B, H*W, 2] 位置proposals,用于encoder中间层监督(如果需要)
            sparse_token_nums: [B] 每个样本实际使用的稀疏token数量
                - 当不同样本有效token数不同时使用
                - BEV场景通常相同,但保留此参数以支持通用情况
                
        稀疏化流程:
            1. Gather: 根据topk_indices提取重要tokens
            2. Compute: 只对K个tokens执行attention + FFN
            3. Scatter: 将更新后的tokens写回原位置
                - 如果sparse_token_nums不为None,每个样本使用不同数量
                - 否则所有样本都使用K个tokens
                如果为None,则执行正常的完整计算
                如果不为None,则只更新top-k个tokens (Sparse DETR模式)
            
        Sparse DETR稀疏化流程:
            对每个encoder layer:
                1. gather稀疏tokens (从[B, L, C]中提取[B, K, C])
                2. 执行layer计算 (attention+FFN只在K个tokens上)
                3. scatter回完整序列 (将[B, K, C]写回[B, L, C])
            
        优化效果:
            - Attention计算: O(L²) → O(K×L)
            - FFN计算: O(L) → O(K)
            - 总体约90%计算节省 (当rho=0.1时)
        """
        # 步骤1: 初始化BEV query和参考点
        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5]-self.pc_range[2], 
            self.num_points_in_pillar, dim='3d', 
            bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', 
            bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs['img_metas'])

        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        # 步骤2: 准备输入格式
        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack(
                [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)

        # 步骤3: 逐层处理 - Sparse DETR稀疏化核心
        use_sparse = topk_indices is not None
        
        # 【Encoder辅助损失 - 语义分割版本】保存中间层预测
        enc_inter_outputs_class = []   # 存储encoder中间层的分类预测 (像素级语义)
        
        for lid, layer in enumerate(self.layers):
            if use_sparse:
                # 【稀疏化路径 - Sparse DETR实现】
                # 3.1 Gather: 提取top-k tokens
                sparse_bev_query = gather_tokens_by_indices(bev_query, topk_indices)
                # sparse_bev_query: [bs, K, C] 例如 [4, 2000, 256]
                
                # 3.2 准备稀疏tokens的位置编码和参考点
                sparse_bev_pos = gather_tokens_by_indices(bev_pos, topk_indices)
                
                # 对于hybird_ref_2d: 无论prev_bev是否存在,hybird_ref_2d都是[bs*2, len_bev, ...]
                # 所以都需要使用doubled indices
                topk_indices_doubled = torch.cat([topk_indices, topk_indices], dim=0)  # [bs*2, K]
                sparse_hybird_ref_2d = gather_tokens_by_indices(
                    hybird_ref_2d.reshape(bs*2, len_bev, -1),
                    topk_indices_doubled
                ).reshape(bs*2, -1, num_bev_level, 2)
                
                # prev_bev的gather取决于是否存在
                if prev_bev is not None:
                    sparse_prev_bev = gather_tokens_by_indices(prev_bev, topk_indices_doubled)
                else:
                    sparse_prev_bev = None
                
                # 对于ref_3d: [bs, num_points_in_pillar, len_bev, 3]
                num_points = ref_3d.shape[1]
                sparse_ref_3d_list = []
                for p in range(num_points):
                    sparse_ref = gather_tokens_by_indices(ref_3d[:, p], topk_indices)  # [bs, K, 3]
                    sparse_ref_3d_list.append(sparse_ref.unsqueeze(1))
                sparse_ref_3d = torch.cat(sparse_ref_3d_list, dim=1)  # [bs, num_points, K, 3]
                
                # 对于reference_points_cam和bev_mask: [num_cam, bs, len_bev, num_points_in_pillar, ...]
                num_cam = reference_points_cam.shape[0]
                # num_points = reference_points_cam.shape[3]
                # reference_points_cam: [num_cam, bs, len_bev, num_points, 2]
                sparse_ref_cam_list = []
                for c in range(num_cam):
                    sparse_ref_cam = gather_tokens_by_indices(
                        reference_points_cam[c].reshape(bs, len_bev, -1),
                        topk_indices
                    ).reshape(bs, -1, num_points, 2)  # test reference_points_cam torch.Size([6, 1, 20000, 4, 2]),topk_indices torch.Size([1, 10000]),sparse_ref_cam torch.Size([1, 40000, 1, 2]);;;train reference_points_cam torch.Size([6, 4, 20000, 4, 2]),topk_indices torch.Size([4, 10000]),sparse_ref_cam torch.Size([4, 10000, 4, 2])
                    sparse_ref_cam_list.append(sparse_ref_cam.unsqueeze(0))
                sparse_ref_points_cam = torch.cat(sparse_ref_cam_list, dim=0)  # [num_cam, bs, K, num_points, 2]
                
                # bev_mask: [num_cam, bs, len_bev, num_points]
                sparse_mask_list = []
                for c in range(num_cam):
                    sparse_mask = gather_tokens_by_indices(
                        bev_mask[c].reshape(bs, len_bev, -1).float(),
                        topk_indices
                    ).reshape(bs, -1, num_points).bool()
                    sparse_mask_list.append(sparse_mask.unsqueeze(0))
                sparse_bev_mask = torch.cat(sparse_mask_list, dim=0)  # [num_cam, bs, K, num_points]
                
                # 3.3 只对稀疏tokens执行layer计算
                # BEVFormerLayer期望batch_first格式: [bs, num_query, C]
                # 稀疏化后: [bs, K, C]
                sparse_bev_query_input = sparse_bev_query  # [bs, K, C]
                sparse_bev_pos_input = sparse_bev_pos  # [bs, K, C]
                
                # 稀疏化时的spatial_shapes处理:
                # - self_attn: 必须使用原始BEV尺寸[bev_h, bev_w]=[200,100]
                #   因为ref_2d是从原始网格gather的,坐标系仍在[0,1]×[0,1]
                #   offset_normalizer需要用原始尺寸归一化,才能正确计算采样位置
                # - cross_attn: 使用图像特征的spatial_shapes=[15,25]
                K = topk_indices.shape[1]
                
                # 参考 Sparse DETR: 使用 tgt 参数实现稀疏 cross-attention
                # tgt: 稀疏tokens做self-attention和作为cross-attention的query
                # query: 完整序列作为cross-attention的src (key/value)
                sparse_output = layer(
                    bev_query,  # 完整序列作为src (用于cross-attention的key/value)
                    key,
                    value,
                    *args,
                    bev_pos=sparse_bev_pos_input,
                    ref_2d=sparse_hybird_ref_2d,
                    ref_3d=sparse_ref_3d,
                    bev_h=bev_h,  # 使用原始BEV尺寸200 (关键!)
                    bev_w=bev_w,  # 使用原始BEV尺寸100 (关键!)
                    spatial_shapes=spatial_shapes,  # cross attention使用图像spatial_shapes
                    level_start_index=level_start_index,
                    reference_points_cam=sparse_ref_points_cam,
                    bev_mask=sparse_bev_mask,
                    prev_bev=sparse_prev_bev,
                    is_sparse=True,  # 标记稀疏模式
                    tgt=sparse_bev_query_input,  # 稀疏tokens (Sparse DETR模式)
                    **kwargs
                )
                
                # 3.4 Scatter: 将更新后的稀疏tokens写回完整序列
                # sparse_output已经是[bs, K, C]格式,不需要permute
                # 支持sparse_token_nums: 不同样本可能使用不同数量的tokens
                if sparse_token_nums is None:
                    # 所有样本都使用K个tokens
                    bev_query = scatter_tokens_by_indices(bev_query, sparse_output, topk_indices)
                else:
                    # 每个样本使用不同数量的tokens (例如因为padding不同)
                    # 注意: BEV场景通常不会遇到这种情况,因为所有样本BEV大小相同
                    # 避免inplace操作,使用list收集后stack
                    updated_queries = []
                    for i in range(bs):
                        valid_k = sparse_token_nums[i]
                        updated_query = scatter_tokens_by_indices(
                            bev_query[i:i+1],  # [1, len_bev, C]
                            sparse_output[i:i+1, :valid_k, :],  # [1, valid_k, C]
                            topk_indices[i:i+1, :valid_k]  # [1, valid_k]
                        )  # [1, len_bev, C]
                        updated_queries.append(updated_query)
                    bev_query = torch.cat(updated_queries, dim=0)  # [bs, len_bev, C]
                
            else:
                # 【正常路径】完整计算
                # BEVFormerLayer期望batch_first格式,bev_query已经是[bs, len_bev, C]
                output = layer(
                    bev_query,
                    key,
                    value,
                    *args,
                    bev_pos=bev_pos,
                    ref_2d=hybird_ref_2d,
                    ref_3d=ref_3d,
                    bev_h=bev_h,
                    bev_w=bev_w,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    reference_points_cam=reference_points_cam,
                    bev_mask=bev_mask,
                    prev_bev=prev_bev,
                    **kwargs
                )
                
                bev_query = output  # output已经是[bs, len_bev, C]
            
            # 保存中间结果
            if self.return_intermediate:
                intermediate.append(bev_query.permute(1, 0, 2))  # 转为[L, bs, C]
            
            # 【Encoder辅助损失 - 语义分割版本】
            # 原理: Encoder输出BEV语义特征,不是实例预测
            # 只需要分类预测 (像素级车道线存在性),不需要坐标预测
            should_generate_aux = self.aux_heads and (
                len(self.layers) == 1 or  # 单层encoder:总是生成
                lid < len(self.layers) - 1  # 多层encoder:排除最后一层
            )
            if should_generate_aux:
                # bev_query: [bs, len_bev, C]
                if cls_branches is not None:
                    # 只使用分类分支生成像素级语义预测
                    # 转换为 (num_query, bs, embed_dims) 格式
                    layer_output = bev_query.permute(1, 0, 2)  # [len_bev, bs, C]
                    
                    # 调用分类分支: 预测每个BEV位置的车道线类别概率
                    output_class = cls_branches[lid](layer_output)  # [len_bev, bs, num_class]
                    
                    # 保存encoder中间层输出 (只保存分类,不需要坐标)
                    enc_inter_outputs_class.append(output_class)

        # 步骤4: 返回结果
        if self.return_intermediate:
            ret = torch.stack(intermediate)
        else:
            # 转回 (num_query, bs, embed_dims) 格式
            ret = bev_query.permute(1, 0, 2)
        
        # 【Encoder辅助损失 - 语义分割版本】如果启用,返回中间层预测
        if self.aux_heads and len(enc_inter_outputs_class) > 0:
            # 返回格式: (encoder_output, enc_inter_class)
            # 不再返回坐标预测,因为使用语义分割损失
            return ret, enc_inter_outputs_class
        else:
            return ret
