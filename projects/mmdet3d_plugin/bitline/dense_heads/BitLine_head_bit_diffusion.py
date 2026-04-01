# Copyright (C) 2024 Xiaomi Corporation.
# Licensed under the Apache License, Version 2.0

"""
CGTopoHeadBitDiffusion: Discrete Bit Diffusion for Lane Topology Prediction

Core Innovation:
- Bit/Bernoulli Diffusion: Discrete diffusion process for binary adjacency matrices
- TopoResNet: Lightweight 2D CNN (no Transformer) to avoid OOM
- Full Graph Training: Train on entire 50×50 graph for train/test consistency

Key Differences from Gaussian Diffusion:
1. Discrete noise process (flip bits) instead of Gaussian noise
2. 2D Conv instead of Transformer (10x faster, 5x less memory)
3. Direct x_0 prediction instead of noise prediction
4. Full graph training (no subgraph sampling)

Author: AI Assistant
Date: 2026-01-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from mmdet.models import HEADS
from mmdet.core import multi_apply
from .BitLine_head import CGTopoHead
from .BitLine_head_diffusion import extract_lane_endpoints, compute_pairwise_geometry


class AttentivePooling(nn.Module):
    """
    自适应注意力池化模块 (Adaptive Attentive Pooling)
    
    核心思想：
    - 让网络自动学习哪些点对拓扑推理最重要
    - 通过注意力机制对 20 个点进行加权求和
    - 重要的点（如拐点、端点）获得更高权重
    
    原理：
    - Content-Adaptive Feature Abstraction（内容自适应特征抽象）
    - 相比 Global Average Pooling，能动态聚焦于最具判别力的几何特征区域
    - 维度保持不变 (256 -> 256)，无需修改后续 ConditionEncoder
    
    Args:
        in_channels: 输入特征维度 (默认: 256)
    """
    def __init__(self, in_channels):
        super(AttentivePooling, self).__init__()
        # 轻量级 Attention 网络
        self.attn_net = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.Tanh(),  # Tanh 激活在 Attention 中常用
            nn.Linear(in_channels // 2, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: [Batch, Num_Lanes, Num_Pts, Channels] e.g. [B, 50, 20, 256]
            
        Returns:
            out: [Batch, Num_Lanes, Channels] e.g. [B, 50, 256]
        """
        # 1. 计算注意力分数
        # attn_scores: [B, N, P, 1]
        attn_scores = self.attn_net(x)
        
        # 2. Softmax 归一化 (在点维度上，确保权重和为 1)
        attn_weights = F.softmax(attn_scores, dim=2)  # [B, N, P, 1]
        
        # 3. 加权求和
        out = (x * attn_weights).sum(dim=2)  # [B, N, C]
        
        return out


class FourierGeometryEncoder(nn.Module):
    """
    基于傅里叶特征的几何编码器 (Fourier Feature Geometry Encoder)
    
    核心思想：
    - 解决"数值/标量"难以被 MLP 学习的问题
    - 使用 sin/cos 将低维数值映射到高维空间（类似 NeRF 的 positional encoding）
    - 让网络能够敏锐地感知位置的微小变化
    
    原理：
    - 对于输入 v ∈ R^d，映射为 [sin(Bv), cos(Bv)] ∈ R^{2k}
    - B 是随机初始化的频率矩阵，控制映射的频率范围
    - sigma 控制频率分布的标准差（类似于 NeRF 中的 L）
    
    参考文献：
    - Fourier Features Let Networks Learn High Frequency Functions (Tancik et al., NeurIPS 2020)
    - NeRF: Representing Scenes as Neural Radiance Fields (Mildenhall et al., ECCV 2020)
    
    Args:
        input_dim: 输入维度（例如：dx, dy, distance = 3）
        embed_dim: 输出嵌入维度（必须是偶数，默认 64）
        sigma: 频率矩阵的标准差（默认 10.0）
    """
    def __init__(self, input_dim=3, embed_dim=64, sigma=10.0):
        super(FourierGeometryEncoder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # 随机初始化频率矩阵 B ~ N(0, sigma^2)
        # 这种映射能让网络更好地学习高频几何细节
        self.register_buffer('B', torch.randn(input_dim, embed_dim // 2) * sigma)

    def forward(self, v):
        """
        Args:
            v: [Batch, N, N, input_dim] 例如: [dx, dy, distance]
            
        Returns:
            out: [Batch, N, N, embed_dim] 傅里叶特征
        """
        # [B, N, N, input_dim] @ [input_dim, embed_dim//2] -> [B, N, N, embed_dim//2]
        projection = torch.matmul(v, self.B)
        
        # sin, cos 变换 -> [B, N, N, embed_dim]
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


class ConditionEncoder(nn.Module):
    """
    改进版 Condition Encoder（使用傅里叶特征映射）
    
    关键改进：
    1. 几何特征升维编码：
       - 旧版：直接使用标量 [distance, angle] -> MLP
       - 新版：[dx, dy, dist] -> Fourier Encoding (64D) -> MLP
    
    2. 解决 Domain Gap：
       - 语义特征：高维、抽象（来自 Transformer Query）
       - 几何特征：低维、具体（物理坐标/距离）
       - 傅里叶编码将几何特征映射到与语义特征相同的表征空间
    
    3. 高频细节学习：
       - 标量输入对于 MLP 难以学习精细的几何变化
       - sin/cos 映射提供多尺度频率信息，类似于多分辨率特征
    
    消融实验对比：
    - use_geo_encoder=False: 仅语义特征（baseline）
    - use_geo_encoder=True: 语义 + 傅里叶几何特征（推荐）
    
    Args:
        query_dim: Query 特征维度（默认 256）
        geo_dim: 几何输入维度（保留参数，实际使用 3: dx, dy, dist）
        cond_dim: 输出条件维度（默认 128）
        use_geo_encoder: 是否使用几何编码器（默认 True）
    """
    
    def __init__(self, query_dim=256, geo_dim=3, cond_dim=128, use_geo_encoder=True):
        super(ConditionEncoder, self).__init__()
        
        self.use_geo_encoder = use_geo_encoder
        
        # ========== 1. 语义分支 ==========
        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim, cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim, cond_dim // 2)
        )
        
        # ========== 2. 几何分支（使用傅里叶编码）==========
        if self.use_geo_encoder:
            self.geo_embed_dim = 64  # 将 3 个数值映射到 64 维
            self.geo_encoding = FourierGeometryEncoder(
                input_dim=3,  # dx, dy, distance
                embed_dim=self.geo_embed_dim,
                sigma=10.0  # 频率范围（可调）
            )
            
            # 将傅里叶特征映射到 cond_dim/4
            self.geo_mlp = nn.Sequential(
                nn.Linear(self.geo_embed_dim, cond_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(cond_dim // 4, cond_dim // 4)
            )
            fusion_input_dim = cond_dim + cond_dim // 4  # query_i + query_j + geo
        else:
            fusion_input_dim = cond_dim  # 仅 query_i + query_j

        # ========== 3. 融合层 ==========
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, query_feat, lane_pts):
        """
        Args:
            query_feat: [B, N, D] Query feature embeddings
            lane_pts: [B, N, P, 2] Lane point coordinates（仅当 use_geo_encoder=True 时使用）
            
        Returns:
            cond: [B, N, N, cond_dim] Pairwise condition features
        """
        B, N, D = query_feat.shape
        
        # ========== 语义特征处理 ==========
        query_embed = self.query_encoder(query_feat)  # [B, N, cond_dim/2]
        query_i = query_embed.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, cond_dim/2]
        query_j = query_embed.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, cond_dim/2]
        
        if self.use_geo_encoder:
            # ========== 几何特征计算（关键修改）==========
            # 1. 提取端点
            # lane_pts: [B, N, P, 2]
            start_pts = lane_pts[:, :, 0, :]   # [B, N, 2] - 起点
            end_pts   = lane_pts[:, :, -1, :]  # [B, N, 2] - 终点
            
            # 2. 计算相对几何关系 (lane_i -> lane_j)
            # 核心思想：lane_j 的起点相对于 lane_i 的终点的位移
            end_i = end_pts.unsqueeze(2)      # [B, N, 1, 2]
            start_j = start_pts.unsqueeze(1)  # [B, 1, N, 2]
            delta = start_j - end_i           # [B, N, N, 2] (dx, dy)
            
            # 3. 计算距离
            dist = torch.norm(delta, dim=-1, keepdim=True)  # [B, N, N, 1]
            
            # 4. 组合原始几何向量 [dx, dy, dist]
            raw_geo = torch.cat([delta, dist], dim=-1)  # [B, N, N, 3]
            
            # 5. 傅里叶编码（数值 -> 高维向量）
            # 这是关键步骤：将物理数值映射到神经网络的表征空间
            geo_emb = self.geo_encoding(raw_geo)  # [B, N, N, 64]
            
            # 6. MLP 变换
            geo_feat = self.geo_mlp(geo_emb)  # [B, N, N, cond_dim/4]
            
            # 7. 拼接所有特征
            cond = torch.cat([query_i, query_j, geo_feat], dim=-1)  # [B, N, N, 160]
        else:
            # 消融版本：仅使用语义特征
            cond = torch.cat([query_i, query_j], dim=-1)  # [B, N, N, 128]
            
        # ========== 融合 ==========
        cond = self.fusion(cond)  # [B, N, N, cond_dim]
        
        return cond


class ResBlock(nn.Module):
    """
    Residual block for TopoResNet.
    
    Standard conv-norm-relu-conv-norm-add pattern with skip connection.
    """
    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super(ResBlock, self).__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(8, channels)  # Group norm works better for small batch
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout2d(dropout)
        self.activation = nn.SiLU()
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        x = self.activation(x)
        return x


class TopoResNet(nn.Module):
    """
    Lightweight 2D CNN for topology denoising (replaces heavy Transformer).
    
    Treats N×N adjacency matrix as a 2D image and applies convolutions.
    Key advantage: O(N²) complexity vs O(N⁴) for Transformer on flattened sequence.
    
    Architecture:
        Input: [B, 1+cond_dim, N, N] (noisy adj + condition features)
        → Conv layers with ResBlocks
        → Output: [B, 1, N, N] (logits for x_0 prediction)
    
    Args:
        input_channels: 1 (noisy adj) + cond_dim (condition features)
        hidden_channels: Hidden dimension for conv layers (default: 128)
        num_layers: Number of ResBlocks (default: 6)
    """
    def __init__(self, input_channels, hidden_channels=128, num_layers=6, dropout=0.1):
        super(TopoResNet, self).__init__()
        
        self.hidden_channels = hidden_channels
        
        # Time embedding (same as diffusion models)
        time_emb_dim = hidden_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_channels * 4),
            nn.SiLU(),
            nn.Linear(hidden_channels * 4, hidden_channels)
        )
        
        # Input projection
        self.input_proj = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        
        # ResBlocks
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_channels, kernel_size=3, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_channels // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1)
        )
    
    def forward(self, x_t, timestep, cond):
        """
        Args:
            x_t: [B, 1, N, N] Noisy adjacency matrix at timestep t
            timestep: [B] Current diffusion timestep
            cond: [B, N, N, cond_dim] Condition features
            
        Returns:
            x_0_logits: [B, 1, N, N] Predicted x_0 logits
        """
        B, _, N, _ = x_t.shape
        
        # ========== Time Embedding ==========
        t_emb = self.get_timestep_embedding(timestep, self.hidden_channels)
        t_emb = self.time_mlp(t_emb)  # [B, hidden_channels]
        
        # ========== Concatenate Input ==========
        # cond: [B, N, N, C] → [B, C, N, N]
        cond = cond.permute(0, 3, 1, 2)
        x = torch.cat([x_t, cond], dim=1)  # [B, 1+C, N, N]
        
        # ========== Input Projection ==========
        x = self.input_proj(x)  # [B, hidden_channels, N, N]
        
        # ========== Add Time Embedding ==========
        # Broadcast time embedding to spatial dimensions
        t_emb = t_emb.view(B, self.hidden_channels, 1, 1)
        x = x + t_emb
        
        # ========== ResBlocks ==========
        for block in self.res_blocks:
            x = block(x)
        
        # ========== Output Projection ==========
        x_0_logits = self.output_proj(x)  # [B, 1, N, N]
        
        return x_0_logits
    
    @staticmethod
    def get_timestep_embedding(timesteps, embedding_dim):
        """Sinusoidal timestep embeddings."""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class GlobalContextBlock(nn.Module):
    """
    简单的全局上下文模块，类似 SE-Block。
    
    通过全局平均池化获取全局信息，然后通过 MLP 生成权重，
    对输入特征进行加权，从而注入全局上下文。
    """
    def __init__(self, channels):
        super(GlobalContextBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


class TopoMLPMixer(nn.Module):
    """
    替换 TopoResNet。
    使用 1x1 卷积处理每个点对，配合 Global Pooling 获取全局信息。
    
    Key Advantages:
    - 1x1 Conv: 只处理每个位置的通道信息，不引入空间依赖假设，符合 Query 无序特性
    - Global Context: 让每个点对感知"整体图"信息，避免 N×N 注意力的 OOM
    - 参数高效: 比 3x3 Conv 参数少 9 倍
    
    Architecture:
        Input: [B, 1+cond_dim, N, N] (noisy adj + condition features)
        → 1x1 Conv + Global Context + Channel Mixing
        → Output: [B, 1, N, N] (logits for x_0 prediction)
    
    Args:
        input_channels: 1 (noisy adj) + cond_dim (condition features)
        hidden_channels: Hidden dimension (default: 128)
        num_layers: Number of mixing layers (default: 4)
    """
    def __init__(self, input_channels, hidden_channels=128, num_layers=4, dropout=0.1):
        super(TopoMLPMixer, self).__init__()
        
        self.hidden_channels = hidden_channels
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.SiLU(),
            nn.Linear(hidden_channels * 4, hidden_channels)
        )
        
        # Input projection (1x1 Conv)
        self.input_proj = nn.Conv2d(input_channels, hidden_channels, 1)
        
        # MLP-Mixer layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                # 1. Channel Mixing (1x1 Conv)
                nn.Conv2d(hidden_channels, hidden_channels, 1),
                nn.GroupNorm(8, hidden_channels),
                nn.SiLU(),
                # 2. Global Context Injection
                # 让每个点对知道"整体图"长什么样
                GlobalContextBlock(hidden_channels),
                nn.Dropout2d(dropout)
            )
            for _ in range(num_layers)
        ])
        
        # Output projection (1x1 Conv)
        self.output_proj = nn.Conv2d(hidden_channels, 1, 1)
    
    def forward(self, x_t, timestep, cond):
        """
        Args:
            x_t: [B, 1, N, N] Noisy adjacency matrix at timestep t
            timestep: [B] Current diffusion timestep
            cond: [B, N, N, cond_dim] Condition features
            
        Returns:
            x_0_logits: [B, 1, N, N] Predicted x_0 logits
        """
        B, _, N, _ = x_t.shape
        
        # ========== Time Embedding ==========
        t_emb = self.get_timestep_embedding(timestep, self.hidden_channels)
        t_emb = self.time_mlp(t_emb).view(B, -1, 1, 1)
        
        # ========== Concatenate Input ==========
        cond = cond.permute(0, 3, 1, 2)
        x = torch.cat([x_t, cond], dim=1)
        
        # ========== Input Projection + Time ==========
        x = self.input_proj(x) + t_emb
        
        # ========== MLP-Mixer Layers with Residual ==========
        for layer in self.layers:
            residual = x
            x = layer(x) + residual
        
        # ========== Output Projection ==========
        x_0_logits = self.output_proj(x)
        
        return x_0_logits
    
    @staticmethod
    def get_timestep_embedding(timesteps, embedding_dim):
        """Sinusoidal timestep embeddings."""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class BitDiffusion(nn.Module):
    """
    Bit/Bernoulli Diffusion Scheduler for binary adjacency matrices.
    
    Discrete diffusion process:
    - Forward: Flip bits with probability (1 - alpha_bar_t)
    - Reverse: Predict original x_0 from noisy x_t
    
    Advantages over Gaussian Diffusion:
    1. Native support for binary data (no need to normalize to [-1, 1])
    2. More interpretable noise process (bit flips vs Gaussian noise)
    3. Direct x_0 prediction (simpler than noise prediction)
    
    Args:
        num_train_timesteps: Training diffusion steps (default: 1000)
        num_inference_steps: Inference steps (default: 20)
        alpha_schedule: Noise schedule type (default: 'cosine')
    """
    def __init__(self, num_train_timesteps=1000, num_inference_steps=20, alpha_schedule='cosine'):
        super(BitDiffusion, self).__init__()
        
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        # ========== Alpha Schedule ==========
        if alpha_schedule == 'linear':
            # Linear schedule: alpha_bar_t decreases linearly
            alphas_cumprod = torch.linspace(1.0, 0.01, num_train_timesteps)
        elif alpha_schedule == 'cosine':
            # Cosine schedule (recommended for bit diffusion)
            steps = num_train_timesteps + 1
            s = 0.008  # Small offset to prevent singularity at t=0
            x = torch.linspace(0, num_train_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_train_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            alphas_cumprod = alphas_cumprod[1:]  # Remove first element
            alphas_cumprod = torch.clip(alphas_cumprod, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown alpha_schedule: {alpha_schedule}")
        
        # Register as buffer
        self.register_buffer('alphas_cumprod', alphas_cumprod)
    
    def q_sample(self, x_0, t):
        """
        Forward diffusion: Add discrete noise to binary adjacency matrix.
        
        Process:
        - With probability alpha_bar_t: keep original value x_0
        - With probability (1 - alpha_bar_t): replace with random bit (0 or 1)
        
        This is equivalent to:
            x_t[i] = x_0[i] with prob alpha_bar_t
            x_t[i] ~ Bernoulli(0.5) with prob (1 - alpha_bar_t)
        
        Args:
            x_0: [B, N, N] Binary adjacency matrix (0 or 1)
            t: [B] Timestep
            
        Returns:
            x_t: [B, N, N] Noisy adjacency matrix
        """
        B, N, _ = x_0.shape
        device = x_0.device
        
        # Get alpha_bar_t for each sample
        alpha_bar_t = self.alphas_cumprod[t].view(B, 1, 1)  # [B, 1, 1]
        
        # Generate random bits
        random_bits = torch.randint(0, 2, (B, N, N), device=device, dtype=x_0.dtype)
        
        # Generate mask: True means keep x_0, False means replace with random
        keep_mask = torch.rand(B, N, N, device=device) < alpha_bar_t
        
        # Apply noise
        x_t = torch.where(keep_mask, x_0, random_bits)
        
        return x_t
    
    def p_sample(self, model, x_t, t, cond, threshold=0.5, temperature=1.0):
        """
        Reverse diffusion: Denoise one step.
        
        Unlike DDPM/DDIM, we directly predict x_0 from x_t.
        Then sample x_{t-1} based on posterior q(x_{t-1} | x_t, x_0).
        
        Args:
            model: Denoising network (TopoResNet)
            x_t: [B, N, N] Noisy adjacency at timestep t
            t: [B] Current timestep
            cond: [B, N, N, cond_dim] Condition features
            threshold: Threshold for binarization (default: 0.5)
            temperature: Temperature scaling (>1: softer, more connections; <1: sharper)
            
        Returns:
            x_t_prev: [B, N, N] Denoised adjacency at timestep t-1
        """
        B, N, _ = x_t.shape
        device = x_t.device
        
        # ========== Predict x_0 ==========
        x_t_input = x_t.unsqueeze(1).float()  # [B, 1, N, N]
        x_0_logits = model(x_t_input, t, cond)  # [B, 1, N, N]
        x_0_logits = x_0_logits.squeeze(1)  # [B, N, N]
        
        # ========== Temperature Scaling ==========
        # Temperature > 1: Softens probabilities, encourages more connections
        # Temperature < 1: Sharpens probabilities, makes model more conservative
        if temperature != 1.0:
            x_0_logits = x_0_logits / temperature
        
        # Convert to probability
        x_0_prob = torch.sigmoid(x_0_logits)  # [B, N, N]
        
        # ========== Sample x_{t-1} ==========
        if t[0] == 0:
            # Final step: deterministic with configurable threshold
            x_t_prev = (x_0_prob > threshold).float()
        else:
            # Compute posterior q(x_{t-1} | x_t, x_0)
            # For bit diffusion, this is a Bernoulli distribution
            alpha_bar_t = self.alphas_cumprod[t].view(B, 1, 1)
            alpha_bar_t_prev = self.alphas_cumprod[t - 1].view(B, 1, 1)
            
            # Posterior probability
            # p(x_{t-1}=1 | x_t, x_0) = alpha_bar_{t-1} * x_0 + (1 - alpha_bar_{t-1}) * 0.5
            posterior_prob = alpha_bar_t_prev * x_0_prob + (1 - alpha_bar_t_prev) * 0.5
            
            # Sample from Bernoulli
            x_t_prev = torch.bernoulli(posterior_prob)
        
        return x_t_prev


@HEADS.register_module()
class CGTopoHeadBitDiffusion(CGTopoHead):
    """
    BitLine Topology Head with Bit Diffusion Model.
    
    Key Improvements over Gaussian Diffusion:
    1. Memory Efficient: 2D CNN (O(N²)) vs Transformer (O(N⁴))
    2. Train/Test Consistent: Full graph training (50×50)
    3. Native Binary Support: Bit diffusion for binary adjacency matrices
    4. Faster: Conv is 10x faster than attention for small N
    
    Training:
        - Full graph (50×50) training with sparse GT
        - BCEWithLogitsLoss on x_0 prediction
        
    Inference:
        - Iterative denoising from random noise
        - 20 steps for generation
    
    Args:
        cond_dim: Condition embedding dimension (default: 128)
        num_train_timesteps: Training diffusion steps (default: 1000)
        num_inference_steps: Inference steps (default: 20)
        hidden_channels: TopoResNet hidden channels (default: 128)
        num_layers: TopoResNet number of layers (default: 6)
        All other args inherited from CGTopoHead
    """
    
    def __init__(self,
                 *args,
                 cond_dim=128,
                 num_train_timesteps=1000,
                 num_inference_steps=20,
                 alpha_schedule='cosine',
                 hidden_channels=128,
                 num_layers=6,
                 inference_threshold=0.9,  # Adjustable threshold for inference
                 inference_temperature=1.0,  # Temperature scaling for inference (>1: more connections)
                 denoising_net_type='TopoResNet',  # NEW: 'TopoResNet' or 'TopoMLPMixer'
                 use_geo_encoder=False,  # 消融实验：是否使用几何编码器（默认False）
                 weight_smooth=0.1,  # NEW: 平滑性损失权重（降低以平衡损失量级）
                 **kwargs):
        
        super(CGTopoHeadBitDiffusion, self).__init__(*args, **kwargs)
        
        # Remove original topology predictor
        del self.vertex_inteact
        del self.lclc_branch
        
        # ========== Adaptive Attentive Pooling ==========
        self.feature_aggregator = AttentivePooling(self.embed_dims)
        
        # ========== Smoothness Loss Weight ==========
        self.weight_smooth = weight_smooth
        
        # ========== Bit Diffusion Components ==========
        # 消融实验：use_geo_encoder 控制是否使用几何特征
        self.cond_encoder = ConditionEncoder(
            query_dim=self.embed_dims,  # 使用注意力池化后的特征维度
            geo_dim=4,
            cond_dim=cond_dim,
            use_geo_encoder=use_geo_encoder  # 传递消融实验参数
        )
        
        # Select denoising network type
        if denoising_net_type == 'TopoResNet':
            self.denoising_net = TopoResNet(
                input_channels=1 + cond_dim,  # noisy adj + condition
                hidden_channels=hidden_channels,
                num_layers=num_layers
            )
        elif denoising_net_type == 'TopoMLPMixer':
            self.denoising_net = TopoMLPMixer(
                input_channels=1 + cond_dim,  # noisy adj + condition
                hidden_channels=hidden_channels,
                num_layers=num_layers
            )
        else:
            raise ValueError(f"Unknown denoising_net_type: {denoising_net_type}. "
                           f"Supported types: 'TopoResNet', 'TopoMLPMixer'")
        
        self.diffusion_scheduler = BitDiffusion(
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            alpha_schedule=alpha_schedule
        )
        
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.inference_threshold = inference_threshold  # Store threshold
        self.inference_temperature = inference_temperature  # Store temperature for scaling
        self._last_cond = None  # GIF可视化用：缓存最后一次推理的条件特征
    
    def forward(self, mlvl_feats, lidar_feat, img_metas, prev_bev=None, only_bev=False):
        """
        Override forward to integrate bit diffusion.
        
        Training: Returns condition features for loss computation
        Inference: Runs iterative denoising loop
        """
        if only_bev:
            return super().forward(mlvl_feats, lidar_feat, img_metas, prev_bev, only_bev)
        
        # ========== Standard DETR Processing ==========
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        
        if self.query_embed_type == 'all_pts':
            object_query_embeds = self.query_embedding.weight.to(dtype)
        elif self.query_embed_type == 'instance_pts':
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)
        
        if self.bev_embedding is not None:
            bev_queries = self.bev_embedding.weight.to(dtype)
            bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                                device=bev_queries.device).to(dtype)
            bev_pos = self.positional_encoding(bev_mask).to(dtype)
        else:
            bev_queries = None
            bev_mask = None
            bev_pos = None

        outputs = self.transformer(
            mlvl_feats,
            lidar_feat,
            bev_queries,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
            prev_bev=prev_bev
        )

        bev_embed, hs, init_reference, inter_references, kp_bev_preds = outputs
        hs = hs.permute(0, 2, 1, 3)
        
        # ========== Process Each Decoder Layer ==========
        outputs_classes = []
        outputs_coords = []
        outputs_pts_coords = []
        outputs_sms = []
        outputs_conds = []
        
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            
            from mmdet.models.utils.transformer import inverse_sigmoid
            reference = inverse_sigmoid(reference)

            outputs_class = self.cls_branches[lvl](
                hs[lvl].view(bs, self.num_vec, self.num_pts_per_vec, -1).mean(2)
            )
            tmp = self.reg_branches[lvl](hs[lvl])
            
            # ========== Extract Features for Diffusion ==========
            # IMPORTANT: NO detach() - allow gradient flow to improve detection head
            # 使用自适应注意力池化替代简单的 mean(2)
            lane_feat_seq = hs[lvl].view(bs, self.num_vec, self.num_pts_per_vec, -1)  # [B, 50, 20, 256]
            vertex_feat = self.feature_aggregator(lane_feat_seq)  # [B, 50, 256]
            
            # Get predicted lane points (denormalize to physical space)
            tmp_for_cond = tmp.clone()
            tmp_for_cond[..., 0:2] += reference[..., 0:2]
            tmp_for_cond = tmp_for_cond.sigmoid()
            
            tmp_pts_normalized = tmp_for_cond.view(bs, self.num_vec, self.num_pts_per_vec, 2)
            from .BitLine_head import denormalize_2d_pts
            tmp_pts_physical = denormalize_2d_pts(
                tmp_pts_normalized.view(bs, -1, 2),
                self.pc_range
            ).view(bs, self.num_vec, self.num_pts_per_vec, 2)
            
            # ========== Compute Condition Features ==========
            cond = self.cond_encoder(vertex_feat, tmp_pts_physical)
            outputs_conds.append(cond)
            
            # ========== Topology Prediction ==========
            if self.training:
                # Training: Don't run denoising, just store condition
                out_sm = torch.zeros((bs, self.num_vec, self.num_vec),
                                    dtype=hs.dtype, device=hs.device)
            else:
                # Inference: Run iterative denoising
                out_sm = self.inference_sampling(cond)
                # Cache cond from the last decoder layer for GIF visualization
                self._last_cond = cond.detach()
            
            # ========== Standard Output Processing ==========
            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            tmp = tmp.sigmoid()

            outputs_coord, outputs_pts_coord = self.transform_box(tmp)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_pts_coords.append(outputs_pts_coord)
            outputs_sms.append(out_sm)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_pts_coords = torch.stack(outputs_pts_coords)
        outputs_sms = torch.stack(outputs_sms)

        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'all_pts_preds': outputs_pts_coords,
            'all_sms_preds': outputs_sms,
            'all_hs': hs,
            'kp_bev_preds': kp_bev_preds,
            'all_conds': outputs_conds if self.training else None,
        }

        return outs
    
    @torch.no_grad()
    def inference_sampling_with_intermediates(self, cond):
        """
        与 generate.py 的 edm_sampler 类似，在每个去噪步存储中间状态。
        
        Args:
            cond: [B, N, N, cond_dim] 条件特征（只取 batch=1）
            
        Returns:
            intermediates: List of [N, N] numpy arrays
                           索引 0: 初始随机噪声
                           索引 1..T: 每个去噪步后的二化邻接矩阵
        """
        B, N, _, _ = cond.shape
        device = cond.device
        
        x_t = torch.randint(0, 2, (B, N, N), device=device, dtype=torch.float32)
        
        timesteps = torch.linspace(
            self.num_train_timesteps - 1, 0, self.num_inference_steps,
            dtype=torch.long, device=device
        )
        
        # 收集初始噪声状态
        intermediates = [x_t[0].cpu().numpy().copy()]
        
        for t_val in timesteps:
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            x_t = self.diffusion_scheduler.p_sample(
                model=self.denoising_net,
                x_t=x_t,
                t=t_batch,
                cond=cond,
                threshold=self.inference_threshold,
                temperature=self.inference_temperature
            )
            intermediates.append(x_t[0].cpu().numpy().copy())
        
        return intermediates  # 共 num_inference_steps+1 帧 [N, N] numpy
    
    @torch.no_grad()
    def inference_sampling(self, cond):
        """
        Iterative denoising for inference.
        
        Args:
            cond: [B, N, N, cond_dim] Condition features
            
        Returns:
            adj_pred: [B, N, N] Binary adjacency matrix
        """
        # print(f"DEBUG: Current Inference Threshold is: {self.inference_threshold}")
        B, N, _, _ = cond.shape
        device = cond.device
        
        # Start from random binary noise
        x_t = torch.randint(0, 2, (B, N, N), device=device, dtype=torch.float32)
        
        # Timestep schedule (reverse order: T-1 -> 0)
        timesteps = torch.linspace(
            self.num_train_timesteps - 1, 0, self.num_inference_steps, dtype=torch.long, device=device
        )
        
        # Iterative denoising
        for t_val in timesteps:
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            x_t = self.diffusion_scheduler.p_sample(
                model=self.denoising_net,
                x_t=x_t,
                t=t_batch,
                cond=cond,
                threshold=self.inference_threshold,  # Use configured threshold
                temperature=self.inference_temperature  # Use temperature scaling
            )
        
        return x_t
    
    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    sms_preds_list,
                    hs_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_sms_list,
                    gt_control_pts_list,
                    gt_bboxes_ignore_list=None):
        """
        Compute regression and classification targets for a batch image.
        
        Modified to return pos_inds and pos_gt_inds for diffusion loss computation.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        # Use multi_apply to call _get_target_single for each sample
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pts_targets_list, pts_weights_list,
         loss_adj_list, loss_beizer_list,
         pos_inds_list, neg_inds_list, pos_gt_inds_list) = multi_apply(
            self._get_target_single, 
            cls_scores_list, bbox_preds_list, pts_preds_list, sms_preds_list, hs_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_sms_list, 
            gt_control_pts_list, gt_bboxes_ignore_list)
        
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                loss_adj_list, loss_beizer_list, 
                num_total_pos, num_total_neg, pos_inds_list, pos_gt_inds_list)
    
    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           sm_pred,
                           hs_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_sms,
                           gt_control_pts,
                           gt_bboxes_ignore=None):
        """
        Compute regression and classification targets for one image.
        
        Returns pos_gt_inds for diffusion loss alignment.
        """
        num_bboxes = bbox_pred.size(0)
        gt_c = gt_bboxes.shape[-1]
        
        # Assignment and sampling
        assign_result, order_index = self.assigner.assign(
            bbox_pred, cls_score, pts_pred,
            gt_bboxes, gt_labels, gt_shifts_pts,
            gt_bboxes_ignore)
        
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_gt_inds = sampling_result.pos_assigned_gt_inds

        # Label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # BBox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # Adjacency matrix targets (for parent's adj loss)
        sm_pred_sub = sm_pred[pos_inds][:, pos_inds]
        sm_target = gt_sms[pos_gt_inds][:, pos_gt_inds].to(torch.float32)
        loss_adj = self.loss_adj(sm_pred_sub, sm_target) * self.weight_adj

        # Bezier control points loss
        pred_control_pts = torch.zeros((len(pos_inds), len(pos_inds), 
                                       self.nums_ctp, 2)).to(pts_pred.device)
        gt_control_pts_sub = gt_control_pts[pos_gt_inds][:, pos_gt_inds]
        hs_pred_permute = hs_pred.view(self.num_vec, self.num_pts_per_vec, -1)[pos_inds]
        
        for i in range(len(sm_target)):
            connection = torch.where(sm_target[i] == 1)[0]
            for c in connection:
                new_line_embed = torch.cat((hs_pred_permute[i], hs_pred_permute[c]))
                beizer_space_embed = torch.matmul(self.inv_B.to(pts_pred.device), new_line_embed)
                control_pts = self.beizer_transform(beizer_space_embed)
                from .BitLine_head import denormalize_2d_pts
                pred_control_pts[i, c] = denormalize_2d_pts(
                    torch.sigmoid(control_pts), self.pc_range)
        
        loss_beizer = self.loss_ctp(pred_control_pts, gt_control_pts_sub,
                                    avg_factor=len(pos_inds))

        # Points targets
        if order_index is None:
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, 
                                        sampling_result.pos_assigned_gt_inds]
        
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                                         pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds, 
                                              assigned_shift, :, :]
        
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights, loss_adj, loss_beizer,
                pos_inds, neg_inds, pos_gt_inds)
    
    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    pts_preds,
                    sms_preds,
                    hs_preds,
                    cond,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_sms_list,
                    gt_control_pts_list,
                    gt_bboxes_ignore_list=None):
        """
        Override loss_single for bit diffusion with FULL GRAPH training.
        
        Key Change:
            - Constructs full 50×50 target matrix (not subgraph)
            - Uses BCEWithLogitsLoss on x_0 prediction
            - No more mixed subgraph sampling (solves train/test inconsistency)
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        sms_preds_list = [sms_preds[i] for i in range(num_imgs)]
        hs_preds_list = [hs_preds[i] for i in range(num_imgs)]
        cond_list = [cond[i] for i in range(num_imgs)]
        
        # Get targets
        cls_reg_targets = self.get_targets(
            cls_scores_list, bbox_preds_list, pts_preds_list, sms_preds_list,
            hs_preds_list, gt_bboxes_list, gt_labels_list, gt_shifts_pts_list,
            gt_sms_list, gt_control_pts_list, gt_bboxes_ignore_list
        )
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list, _, loss_beizer_list,
         num_total_pos, num_total_neg, pos_inds_list, pos_gt_inds_list) = cls_reg_targets
        
        # ========== Bit Diffusion Loss (FULL GRAPH) ==========
        loss_adj_list = []
        
        for i in range(len(cond_list)):
            cond_i = cond_list[i]  # [N, N, cond_dim]
            gt_sms = gt_sms_list[i]  # [num_gt, num_gt]
            pos_inds = pos_inds_list[i]  # [P]
            pos_gt_inds = pos_gt_inds_list[i]  # [P]
            
            N = cond_i.shape[0]  # Total queries (50)
            P = len(pos_inds)
            
            # Skip if no positive samples
            if P == 0:
                continue
            
            # ========== Step 1: Build FULL Graph Target ==========
            # Initialize full graph with all zeros (background)
            target_full = torch.zeros(N, N, dtype=torch.float32, device=cond_i.device)
            # [N, N]
            
            # Fill matched samples' topology
            if P > 0:
                # Extract GT subgraph
                gt_adj_sub = gt_sms[pos_gt_inds][:, pos_gt_inds].float()  # [P, P]
                
                # Use meshgrid indexing to fill
                idx_i = pos_inds.unsqueeze(1).expand(P, P)  # [P, P]
                idx_j = pos_inds.unsqueeze(0).expand(P, P)  # [P, P]
                
                # Fill into full graph
                target_full[idx_i, idx_j] = gt_adj_sub
            
            # Now target_full is:
            # - 1 at (pos_i, pos_j) if GT says they're connected
            # - 0 everywhere else (negative samples + unconnected positive pairs)
            
            # ========== Step 2: Sample Random Timestep ==========
            t = torch.randint(0, self.num_train_timesteps, (1,), device=cond_i.device)
            
            # ========== Step 3: Forward Diffusion (Add Noise) ==========
            noisy_adj = self.diffusion_scheduler.q_sample(
                target_full.unsqueeze(0), t)  # [1, N, N]
            
            # ========== Step 4: Predict x_0 ==========
            noisy_adj_input = noisy_adj.unsqueeze(1)  # [1, 1, N, N]
            x_0_logits = self.denoising_net(
                noisy_adj_input, t, cond_i.unsqueeze(0))  # [1, 1, N, N]
            x_0_logits = x_0_logits.squeeze(1).squeeze(0)  # [N, N]
            
            # ========== Step 5: BCEWithLogitsLoss with Class Balancing ==========
            # Direct prediction loss on x_0
            # Use pos_weight to handle class imbalance (positive:negative ≈ 1:2400)
            num_pos = target_full.sum()
            num_neg = target_full.numel() - num_pos
            # pos_weight = num_neg / (num_pos + 1e-6)
            pos_weight = (num_neg / (num_pos + 1e-6)).clamp(max=50.0)  # Limit max weight
            # pos_weight = (num_neg / (num_pos + 1e-6)).clamp(min=10.0, max=150.0)  # only for timestep500
            
            loss_adj = F.binary_cross_entropy_with_logits(
                x_0_logits, 
                target_full, 
                pos_weight=pos_weight, # 关键参数！
                reduction='mean'
            )

            loss_adj_list.append(loss_adj * self.weight_adj)
        
        loss_adj = sum(loss_adj_list) / max(len(loss_adj_list), 1)
        
        # ========== Other Losses (Same as Original) ==========
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)
        
        loss_beizer = sum(loss_beizer_list) / len(loss_beizer_list)
        
        # Classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            from mmdet.core import reduce_mean
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        from mmdet.core import reduce_mean
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        
        # BBox loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        from .BitLine_head import normalize_2d_bbox
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :4], normalized_bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4], avg_factor=num_total_pos)
        
        # Points loss
        from .BitLine_head import normalize_2d_pts, denormalize_2d_pts
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        
        if self.num_pts_per_vec != self.num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0, 2, 1)
            pts_preds = F.interpolate(pts_preds, size=(self.num_pts_per_gt_vec),
                                    mode='linear', align_corners=True)
            pts_preds = pts_preds.permute(0, 2, 1).contiguous()
        
        loss_pts = self.loss_pts(
            pts_preds[isnotnan, :, :], normalized_pts_targets[isnotnan, :, :],
            pts_weights[isnotnan, :, :], avg_factor=num_total_pos)
        
        # Direction loss
        dir_weights = pts_weights[:, :-self.dir_interval, 0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:, self.dir_interval:, :] - \
                                denormed_pts_preds[:, :-self.dir_interval, :]
        pts_targets_dir = pts_targets[:, self.dir_interval:, :] - \
                         pts_targets[:, :-self.dir_interval, :]
        
        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan, :, :], pts_targets_dir[isnotnan, :, :],
            dir_weights[isnotnan, :], avg_factor=num_total_pos)
        
        # IoU loss
        from .BitLine_head import denormalize_2d_bbox
        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        loss_iou = self.loss_iou(
            bboxes[isnotnan, :4], bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4], avg_factor=num_total_pos)
        
        # ========== NEW: Smoothness Loss (平滑性损失) ==========
        # 目的：减少车道线的锤齿和抖动，让线变得顺滑
        # 原理：通过二阶差分衡量曲率变化，如果线是直的或平滑弯曲，二阶差分接近 0
        # 只对正样本（匹配到GT的车道）计算平滑性损失
        pos_pts_preds = denormed_pts_preds[isnotnan]  # [N_pos, Num_Pts, 2]
        
        if pos_pts_preds.shape[0] > 0 and pos_pts_preds.shape[1] >= 3:
            # 1. 计算一阶差分 (向量): v_i = p_{i+1} - p_i
            # shape: [N_pos, Num_Pts-1, 2]
            diff_1 = pos_pts_preds[:, 1:, :] - pos_pts_preds[:, :-1, :]
            
            # 2. 计算二阶差分 (加速度/曲率): a_i = v_{i+1} - v_i
            # shape: [N_pos, Num_Pts-2, 2]
            diff_2 = diff_1[:, 1:, :] - diff_1[:, :-1, :]
            
            # 3. 最小化二阶差分 (强迫线变直或平滑变弯)
            # 使用 L2 范数计算每个点的曲率变化大小
            loss_smooth = diff_2.norm(dim=-1).mean() * self.weight_smooth
        else:
            loss_smooth = torch.tensor(0.0, device=pts_preds.device)
        
        from mmcv.utils import TORCH_VERSION, digit_version
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)
            loss_adj = torch.nan_to_num(loss_adj)
            loss_smooth = torch.nan_to_num(loss_smooth)
        
        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir, loss_adj, loss_beizer, loss_smooth
    
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """
        Override loss to pass condition features to loss_single.
        """
        assert gt_bboxes_ignore is None
        
        gt_vecs_list = copy.deepcopy(gt_bboxes_list)
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_pts_preds = preds_dicts['all_pts_preds']
        all_sms_preds = preds_dicts['all_sms_preds']
        all_hs_preds = preds_dicts['all_hs']
        all_conds = preds_dicts['all_conds']
        kp_bev_preds = preds_dicts['kp_bev_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [gt_bboxes.bbox.to(device) for gt_bboxes in gt_vecs_list]
        gt_shifts_pts_list = [gt_bboxes.fixed_num_sampled_points_ambiguity.to(device)
                             for gt_bboxes in gt_vecs_list]
        gt_sms_list = [torch.from_numpy(gt_bboxes.adj_matrix).to(device)
                      for gt_bboxes in gt_vecs_list]
        gt_control_pts_list = [gt_bboxes.get_beizer_control_pts.to(device)
                              for gt_bboxes in gt_vecs_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        all_gt_sms_list = [gt_sms_list for _ in range(num_dec_layers)]
        all_gt_control_pts_list = [gt_control_pts_list for _ in range(num_dec_layers)]

        # Compute losses for each layer
        from mmdet.core import multi_apply
        losses_cls, losses_bbox, losses_iou, losses_pts, losses_dir, losses_adj, losses_beizer, losses_smooth = multi_apply(
            self.loss_single,
            all_cls_scores, all_bbox_preds, all_pts_preds, all_sms_preds,
            all_hs_preds, all_conds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_shifts_pts_list,
            all_gt_sms_list, all_gt_control_pts_list,
            all_gt_bboxes_ignore_list
        )

        loss_dict = dict()

        # Keypoint loss
        from .BitLine_head import _neg_loss
        gt_bev_kp_list = [self.get_bev_keypoint(gt_bboxes).to(device)
                         for gt_bboxes in gt_vecs_list]
        gt_bev_kp = torch.stack(gt_bev_kp_list)
        loss_dict['loss_kp'] = _neg_loss(kp_bev_preds, gt_bev_kp, weights=2) * self.weight_kp

        # Last layer losses
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_pts'] = losses_pts[-1]
        loss_dict['loss_dir'] = losses_dir[-1]
        loss_dict['loss_adj'] = losses_adj[-1]
        loss_dict['loss_beizer'] = losses_beizer[-1]
        loss_dict['loss_smooth'] = losses_smooth[-1]  # NEW: 平滑性损失

        # Intermediate layer losses
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i, loss_dir_i, loss_adj_i, loss_beizer_i, loss_smooth_i in zip(
                losses_cls[:-1], losses_pts[:-1], losses_dir[:-1],
                losses_adj[:-1], losses_beizer[:-1], losses_smooth[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
            loss_dict[f'd{num_dec_layer}.loss_adj'] = loss_adj_i
            loss_dict[f'd{num_dec_layer}.loss_beizer'] = loss_beizer_i
            loss_dict[f'd{num_dec_layer}.loss_smooth'] = loss_smooth_i  # NEW
            num_dec_layer += 1
        
        return loss_dict, None, None
