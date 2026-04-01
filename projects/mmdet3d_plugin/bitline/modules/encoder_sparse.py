# Copyright (C) 2024 Xiaomi Corporation.
# Sparse DETR Implementation - Sparse BEV Encoder
# 在BEV特征上实现稀疏化,只更新重要的tokens

import torch
import torch.nn as nn
import numpy as np
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.runner.base_module import BaseModule
from .encoder import LSSTransform  # 继承原始的LSSTransform
from .mask_predictor_sparse import MaskPredictor


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class LSSTransformSparse(LSSTransform):
    """
    稀疏化的LSS Transform - Sparse DETR核心组件
    
    功能:
        在BEV特征提取后,使用DAM预测器选择重要的tokens进行后续处理
        
    新增参数:
        rho: 稀疏率,保留token的比例 (例如0.1表示保留10%的tokens)
        mask_predictor_dim: DAM预测器的隐藏层维度
        enable_sparse: 是否启用稀疏化 (训练时动态控制)
    """
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 feat_down_sample,
                 pc_range,
                 voxel_size,
                 dbound,
                 downsample=1,
                 rho=0.1,  # 稀疏率: 保留10%的tokens
                 mask_predictor_dim=256,  # DAM预测器隐藏维度
                 **kwargs):
        super(LSSTransformSparse, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            feat_down_sample=feat_down_sample,
            pc_range=pc_range,
            voxel_size=voxel_size,
            dbound=dbound,
            downsample=downsample,
        )
        
        # 稀疏化参数
        self.rho = rho
        self.enable_sparse = rho > 0  # 只有rho>0时才启用稀疏化
        
        # DAM预测器: 用于预测每个BEV token的重要性
        if self.enable_sparse:
            self.mask_predictor = MaskPredictor(
                in_dim=out_channels,
                h_dim=mask_predictor_dim
            )
        else:
            self.mask_predictor = None
            
        # 用于存储预测的mask(训练时用于计算DAM损失)
        self.backbone_mask_prediction = None
        self.backbone_topk_proposals = None

    @force_fp32()
    def forward(self, images, img_metas, return_mask_pred=False):
        """
        前向传播 - 支持稀疏化的BEV特征提取
        
        Args:
            images: 输入图像 [B, N, C, H, W]
            img_metas: 图像元信息
            return_mask_pred: 是否返回mask预测结果(训练时为True)
            
        Returns:
            bev_embed: BEV特征 [B, H*W, C]
            如果return_mask_pred=True,还会返回:
                - backbone_mask_prediction: token重要性分数 [B, H*W]
                - backbone_topk_proposals: top-k indices [B, topk_num]
                
        稀疏化流程:
            1. 正常提取BEV特征 (继承自LSSTransform)
            2. 使用DAM预测器预测每个token的重要性
            3. 选择top-k个重要的tokens
            4. 返回完整特征 + mask预测信息(供后续使用)
        """
        # 步骤1: 调用父类方法,提取完整的BEV特征
        # bev_feat: [B, C, H, W]
        bev_feat = super().forward(images, img_metas)
        
        # 步骤2: 转换为token序列格式
        bs, c, h, w = bev_feat.shape
        bev_embed = bev_feat.view(bs, c, -1).permute(0, 2, 1).contiguous()
        # bev_embed: [B, H*W, C]  例如: [2, 20000, 256]
        
        # 步骤3: 如果启用稀疏化,使用DAM预测器
        if self.enable_sparse and return_mask_pred:
            # 预测每个token的重要性分数
            backbone_mask_prediction = self.mask_predictor(bev_embed)  # [B, H*W, 1]
            backbone_mask_prediction = backbone_mask_prediction.squeeze(-1)  # [B, H*W]
            
            # 计算要保留的token数量
            total_tokens = h * w
            sparse_token_nums = int(total_tokens * self.rho)  # 例如: 20000 * 0.1 = 2000
            
            # Top-K选择: 选出最重要的tokens
            # topk返回: (values, indices)
            _, backbone_topk_proposals = torch.topk(
                backbone_mask_prediction,
                k=sparse_token_nums,
                dim=1
            )  # backbone_topk_proposals: [B, sparse_token_nums]
            
            # 保存结果供后续使用
            self.backbone_mask_prediction = backbone_mask_prediction
            self.backbone_topk_proposals = backbone_topk_proposals
            
            return bev_embed, backbone_mask_prediction, backbone_topk_proposals
        else:
            # 不启用稀疏化或推理阶段
            self.backbone_mask_prediction = None
            self.backbone_topk_proposals = None
            return bev_embed

    def get_mask_prediction(self):
        """
        获取最近一次的mask预测结果
        用于计算DAM损失
        """
        return self.backbone_mask_prediction

    def get_topk_proposals(self):
        """
        获取最近一次的top-k indices
        用于稀疏化encoder
        """
        return self.backbone_topk_proposals


# 辅助函数: 根据top-k indices进行gather操作
def gather_tokens_by_indices(tokens, indices):
    """
    从完整的tokens中提取top-k个tokens
    
    Args:
        tokens: [B, L, C] - 完整的token序列
        indices: [B, K] - top-k的索引
        
    Returns:
        selected_tokens: [B, K, C] - 选中的tokens
    """
    B, L, C = tokens.shape
    K = indices.shape[1]
    
    # 扩展indices以匹配特征维度
    indices_expanded = indices.unsqueeze(-1).expand(B, K, C)  # [B, K, C]
    
    # 使用gather提取
    selected_tokens = torch.gather(tokens, 1, indices_expanded)  # [B, K, C]
    
    return selected_tokens


# 辅助函数: 将更新后的tokens scatter回原位置
def scatter_tokens_by_indices(full_tokens, updated_tokens, indices):
    """
    将更新后的稀疏tokens写回完整的token序列
    
    Args:
        full_tokens: [B, L, C] - 完整的token序列
        updated_tokens: [B, K, C] - 更新后的稀疏tokens
        indices: [B, K] - top-k的索引
        
    Returns:
        output_tokens: [B, L, C] - 更新后的完整序列
        
    工作原理:
        只更新indices指定位置的tokens,其余位置保持不变
        这是Sparse DETR的关键: 90%的tokens不更新,节省计算量
    """
    B, L, C = full_tokens.shape
    K = indices.shape[1]
    
    # 复制完整tokens
    output_tokens = full_tokens.clone()
    
    # 扩展indices以匹配特征维度
    indices_expanded = indices.unsqueeze(-1).expand(B, K, C)  # [B, K, C]
    
    # 使用scatter更新指定位置
    output_tokens = output_tokens.scatter(1, indices_expanded, updated_tokens)
    
    return output_tokens
