# Copyright (C) 2024 Xiaomi Corporation.
# Sparse DETR Implementation - DAM (Decoder Attention Map) Predictor
# 用于预测encoder中哪些tokens最重要,从而实现稀疏化

import torch
import torch.nn as nn


class MaskPredictor(nn.Module):
    """
    DAM预测器 - Sparse DETR的核心组件
    
    功能:
        预测每个encoder token的重要性分数,用于Top-K选择
        
    工作流程:
        1. 输入: encoder特征 [B, L, C] (例如: [2, 20000, 256])
        2. Layer1: LayerNorm + Linear + GELU
        3. 特征分离: 局部特征(z_local)和全局特征(z_global)
        4. 全局特征: 对所有tokens取平均,代表全局上下文
        5. 融合: 将局部和全局特征拼接
        6. Layer2: MLP预测重要性分数
        7. 输出: 重要性分数 [B, L, 1]
        
    参数:
        in_dim: 输入特征维度 (通常是256)
        h_dim: 隐藏层维度 (通常是256)
    """
    
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        
        # 第一层: 特征变换和激活
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim),  # 归一化输入特征
            nn.Linear(in_dim, h_dim),  # 线性变换
            nn.GELU()  # 平滑的非线性激活
        )
        
        # 第二层: 多层感知机预测重要性分数
        # 逐渐降维: h_dim -> h_dim//2 -> h_dim//4 -> 1
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1)  # 最终输出单个分数
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: encoder特征 [B, L, C]
               - B: batch size
               - L: token数量 (如BEV特征展平后的长度)
               - C: 特征维度
               
        Returns:
            out: 重要性分数 [B, L, 1]
                 - 每个token一个分数,用于后续Top-K选择
                 
        示例:
            输入: [2, 20000, 256]
            layer1输出: [2, 20000, 256]
            分离后: z_local [2, 20000, 128], z_global [2, 1, 128]
            z_global扩展: [2, 20000, 128]
            拼接: [2, 20000, 256]
            输出: [2, 20000, 1]
        """
        # 第一层变换
        z = self.layer1(x)  # [B, L, h_dim]
        
        # 分离局部和全局特征
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        # z_local: [B, L, h_dim//2] - 保留每个token的局部特征
        # z_global: [B, L, h_dim//2] - 用于计算全局上下文
        
        # 计算全局上下文: 对所有tokens取平均
        z_global = z_global.mean(dim=1, keepdim=True)  # [B, 1, h_dim//2]
        
        # 将全局特征广播到所有tokens
        z_global = z_global.expand(-1, z_local.shape[1], -1)  # [B, L, h_dim//2]
        
        # 融合局部和全局信息
        z = torch.cat([z_local, z_global], dim=-1)  # [B, L, h_dim]
        
        # 预测重要性分数
        out = self.layer2(z)  # [B, L, 1]
        
        return out


def build_mask_predictor(in_dim=256, h_dim=256):
    """
    构建MaskPredictor的工厂函数
    
    Args:
        in_dim: 输入维度
        h_dim: 隐藏层维度
        
    Returns:
        MaskPredictor实例
    """
    return MaskPredictor(in_dim, h_dim)
