# Copyright (C) 2024 Xiaomi Corporation.
# Licensed under the Apache License, Version 2.0

"""
MapDiffusionHead: Conditional Latent Diffusion Model for Lane Topology Prediction

Core Innovation:
- Treats N×N adjacency matrix prediction as an image generation task
- Uses flow alignment score to resolve parallel lane ambiguity
- DDIM sampling for efficient inference (20 steps vs 1000 training steps)

Architecture:
1. ConditionEncoder: Computes pairwise geometric features (alignment + distance)
2. DenoisingNet: Relation Transformer for global topology reasoning
3. GaussianDiffusion: Standard DDPM training + DDIM inference

Author: AI Assistant
Date: 2026-01-07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import copy
from mmdet.models import HEADS
from .BitLine_head import CGTopoHead


def extract_lane_endpoints(lane_pts):
    """
    Extract directional endpoints from lane point sequences.
    
    Args:
        lane_pts: [B, N, num_pts, 2] Lane point coordinates (x, y)
        
    Returns:
        start_pts: [B, N, 2] Starting points
        end_pts: [B, N, 2] Ending points
        direction_vec: [B, N, 2] Normalized direction vectors (start -> end)
    """
    B, N, num_pts, _ = lane_pts.shape
    
    # Extract first and last points
    start_pts = lane_pts[:, :, 0, :]  # [B, N, 2]
    end_pts = lane_pts[:, :, -1, :]   # [B, N, 2]
    
    # Compute direction vector and normalize
    direction_vec = end_pts - start_pts  # [B, N, 2]
    norm = torch.norm(direction_vec, dim=-1, keepdim=True) + 1e-6  # [B, N, 1]
    direction_vec = direction_vec / norm  # [B, N, 2]
    
    return start_pts, end_pts, direction_vec


def compute_pairwise_geometry(lane_pts):
    """
    Compute pairwise geometric features for all lane pairs (fully vectorized).
    
    This is the KEY to resolving parallel lane ambiguity:
    - Flow alignment score distinguishes longitudinal vs lateral connections
    - Euclidean distance provides basic spatial context
    
    Args:
        lane_pts: [B, N, num_pts, 2] Lane point coordinates (physical space, in meters)
        
    Returns:
        geo_features: [B, N, N, 4] Pairwise geometric feature matrix:
            - [0]: Euclidean distance between lane_i.end and lane_j.start
            - [1]: Flow alignment score (cosine similarity)
            - [2]: Delta x
            - [3]: Delta y
    """
    B, N, num_pts, _ = lane_pts.shape
    
    # Extract endpoints and directions
    start_pts, end_pts, direction_vec = extract_lane_endpoints(lane_pts)
    # start_pts: [B, N, 2], end_pts: [B, N, 2], direction_vec: [B, N, 2]
    
    # ========== Step 1: Compute Connection Vectors ==========
    # Broadcast to compute all pairwise connections: end_i -> start_j
    end_i = end_pts.unsqueeze(2)      # [B, N, 1, 2]
    start_j = start_pts.unsqueeze(1)  # [B, 1, N, 2]
    
    connection_vec = start_j - end_i  # [B, N, N, 2] (i->j connection vector)
    
    # ========== Step 2: Euclidean Distance ==========
    euclidean_dist = torch.norm(connection_vec, dim=-1, keepdim=True)  # [B, N, N, 1]
    
    # ========== Step 3: Flow Alignment Score (CRITICAL) ==========
    # This is the key to solving parallel lane ambiguity!
    # Computes: cos(angle) between direction_i and connection_ij
    # - alignment ≈ 1.0: perfect alignment (longitudinal successor)
    # - alignment ≈ 0.0: perpendicular (lateral neighbor)
    # - alignment < 0.0: opposite direction (should be suppressed)
    
    direction_i = direction_vec.unsqueeze(2)  # [B, N, 1, 2]
    
    # Normalize connection vectors
    connection_norm = torch.norm(connection_vec, dim=-1, keepdim=True) + 1e-6  # [B, N, N, 1]
    connection_vec_normalized = connection_vec / connection_norm  # [B, N, N, 2]
    
    # Cosine similarity (dot product of normalized vectors)
    alignment_score = (direction_i * connection_vec_normalized).sum(dim=-1, keepdim=True)
    # [B, N, N, 1]
    
    # ========== Step 4: Concatenate All Features ==========
    geo_features = torch.cat([
        euclidean_dist,      # [B, N, N, 1]
        alignment_score,     # [B, N, N, 1]
        connection_vec       # [B, N, N, 2]
    ], dim=-1)  # [B, N, N, 4]
    
    return geo_features


class ConditionEncoder(nn.Module):
    """
    Condition Encoder: Constructs N×N pairwise feature matrix for diffusion conditioning.
    
    Key Design:
    - Combines semantic features (query_feat) with geometric features (pts_preds)
    - Flow alignment score is the CORE feature for resolving ambiguity
    - Fully vectorized (no Python loops)
    
    Args:
        query_dim: Dimension of query features (e.g., 256)
        geo_dim: Dimension of geometric features (default: 4)
        cond_dim: Output condition dimension (default: 128)
    """
    
    def __init__(self, query_dim=256, geo_dim=4, cond_dim=128):
        super(ConditionEncoder, self).__init__()
        
        # Semantic feature encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim, cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim, cond_dim // 2)
        )
        
        # Geometric feature encoder
        self.geo_encoder = nn.Sequential(
            nn.Linear(geo_dim, cond_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim // 4, cond_dim // 4)
        )
        
        # Fusion layer
        # Input: [query_i: cond_dim/2, query_j: cond_dim/2, geo: cond_dim/4]
        # Total: cond_dim/2 + cond_dim/2 + cond_dim/4 = 1.25*cond_dim
        fusion_input_dim = cond_dim + cond_dim // 4  # 128 + 32 = 160
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, query_feat, lane_pts):
        """
        Args:
            query_feat: [B, N, D] Query feature embeddings
            lane_pts: [B, N, P, 2] Lane point coordinates (physical space)
            
        Returns:
            cond: [B, N, N, cond_dim] Pairwise condition features
        """
        B, N, D = query_feat.shape
        
        # ========== Step 1: Encode Query Features ==========
        query_embed = self.query_encoder(query_feat)  # [B, N, cond_dim/2]
        
        # ========== Step 2: Compute Geometric Features ==========
        geo_features = compute_pairwise_geometry(lane_pts)  # [B, N, N, 4]
        geo_embed = self.geo_encoder(geo_features)  # [B, N, N, cond_dim/4]
        
        # ========== Step 3: Broadcast Semantic Features to N×N ==========
        # Construct pairwise semantic features: [query_i, query_j]
        query_i = query_embed.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, cond_dim/2]
        query_j = query_embed.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, cond_dim/2]
        
        # ========== Step 4: Concatenate All Features ==========
        # Concat semantic + geometric features (NO information loss)
        # [query_i: 64, query_j: 64, geo: 32] → 160 dimensions
        cond = torch.cat([query_i, query_j, geo_embed], dim=-1)  # [B, N, N, 160]
        
        # ========== Step 5: Fusion ==========
        # Learnable fusion: 160 → 128
        # Network automatically learns optimal balance between semantic & geometric
        cond = self.fusion(cond)  # [B, N, N, cond_dim]
        
        return cond


class RelationTransformer(nn.Module):
    """
    Relation Transformer: Denoising network for adjacency matrix.
    
    Treats N×N adjacency matrix as a sequence of length N² and applies:
    - Self-Attention: Captures global topology structure
    - Cross-Attention: Injects geometric/semantic conditions
    
    Args:
        d_model: Hidden dimension (default: 128)
        nhead: Number of attention heads (default: 4)
        num_layers: Number of transformer layers (default: 3)
    """
    
    def __init__(self, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super(RelationTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Time embedding (sinusoidal encoding)
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Input projection: noisy adjacency matrix -> embedding
        self.input_proj = nn.Linear(1, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Condition cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection: embedding -> noise prediction
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, noisy_adj, timestep, cond):
        """
        Args:
            noisy_adj: [B, N, N, 1] Noisy adjacency matrix at timestep t
            timestep: [B] Current diffusion timestep
            cond: [B, N, N, D] Condition features
            
        Returns:
            noise_pred: [B, N, N, 1] Predicted noise
        """
        B, N, _, _ = noisy_adj.shape
        
        # ========== Step 1: Time Embedding ==========
        # Sinusoidal positional encoding for timestep
        t_emb = self.get_timestep_embedding(timestep, self.d_model)  # [B, d_model]
        t_emb = self.time_embed(t_emb)  # [B, d_model]
        
        # ========== Step 2: Flatten N×N to Sequence ==========
        # Treat adjacency matrix as sequence of N² tokens
        adj_flat = noisy_adj.view(B, N * N, 1)  # [B, N², 1]
        cond_flat = cond.view(B, N * N, -1)  # [B, N², D]
        
        # ========== Step 3: Input Projection + Time Conditioning ==========
        x = self.input_proj(adj_flat)  # [B, N², d_model]
        x = x + t_emb.unsqueeze(1)  # Broadcast time embedding
        
        # ========== Step 4: Self-Attention (Global Topology) ==========
        x = self.transformer(x)  # [B, N², d_model]
        
        # ========== Step 5: Cross-Attention (Condition Injection) ==========
        x, _ = self.cross_attn(
            query=x,              # [B, N², d_model]
            key=cond_flat,        # [B, N², D]
            value=cond_flat       # [B, N², D]
        )  # [B, N², d_model]
        
        # ========== Step 6: Output Projection ==========
        noise_pred = self.output_proj(x)  # [B, N², 1]
        noise_pred = noise_pred.view(B, N, N, 1)  # [B, N, N, 1]
        
        return noise_pred
    
    @staticmethod
    def get_timestep_embedding(timesteps, embedding_dim):
        """
        Sinusoidal timestep embeddings (same as in original Diffusion models).
        
        Args:
            timesteps: [B] Timestep values
            embedding_dim: Dimension of output embedding
            
        Returns:
            emb: [B, embedding_dim] Time embeddings
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [B, embedding_dim]
        
        if embedding_dim % 2 == 1:  # Zero pad if odd
            emb = F.pad(emb, (0, 1))
        
        return emb


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Scheduler: Handles forward (noising) and reverse (denoising) processes.
    
    Training: Standard DDPM with T=1000 steps
    Inference: DDIM with T_inf=20 steps for efficiency
    
    Args:
        num_train_timesteps: Training diffusion steps (default: 1000)
        num_inference_steps: Inference steps for DDIM (default: 20)
        beta_schedule: Noise schedule type (default: 'linear')
    """
    
    def __init__(self, num_train_timesteps=1000, num_inference_steps=20, beta_schedule='linear'):
        super(GaussianDiffusion, self).__init__()
        
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        # ========== Noise Schedule ==========
        if beta_schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, num_train_timesteps)
        elif beta_schedule == 'cosine':
            # Improved cosine schedule (https://arxiv.org/abs/2102.09672)
            steps = num_train_timesteps + 1
            s = 0.008
            x = torch.linspace(0, num_train_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_train_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        # Precompute useful constants
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register as buffers (not parameters, but saved in state_dict)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Precompute sqrt values for efficiency
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: Add noise to clean data.
        
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        
        Args:
            x_0: [B, N, N, 1] Clean adjacency matrix
            t: [B] Timestep
            noise: [B, N, N, 1] Optional pre-generated noise
            
        Returns:
            x_t: [B, N, N, 1] Noisy adjacency matrix at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        
        return x_t
    
    def ddim_step(self, model, x_t, t, t_prev, cond, eta=0.0):
        """
        DDIM reverse step: Denoise from x_t to x_{t-1}.
        
        DDIM formula (deterministic when eta=0):
        x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1 - alpha_{t-1} - sigma_t^2) * pred_eps + sigma_t * noise
        
        Args:
            model: Denoising network
            x_t: [B, N, N, 1] Noisy adjacency at timestep t
            t: [B] Current timestep
            t_prev: [B] Previous timestep (for DDIM skip)
            cond: [B, N, N, D] Condition features
            eta: DDIM stochasticity parameter (0=deterministic)
            
        Returns:
            x_t_prev: [B, N, N, 1] Denoised adjacency at timestep t_prev
        """
        # Predict noise
        noise_pred = model(x_t, t, cond)  # [B, N, N, 1]
        
        # Extract alpha values
        alpha_t = self.alphas_cumprod[t][:, None, None, None]
        alpha_t_prev = self.alphas_cumprod[t_prev][:, None, None, None]
        
        # Predict x_0 from x_t and noise
        pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        
        # Clip to valid range (adjacency values should be in [0, 1] after training)
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        
        # Compute direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_t_prev) * noise_pred
        
        # DDIM reverse step
        x_t_prev = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
        
        return x_t_prev


@HEADS.register_module()
class CGTopoHeadDiffusion(CGTopoHead):
    """
    BitLine Topology Head with Conditional Diffusion Model.
    
    Replaces the standard MLP/GNN topology predictor with a generative diffusion model
    that can:
    1. Leverage global context through iterative denoising
    2. Resolve parallel lane ambiguity via flow alignment conditioning
    3. Generate more natural, structurally consistent topologies
    
    Key Design:
    - Training: Standard DDPM loss (MSE between predicted and actual noise)
    - Inference: DDIM sampling (20 steps) for efficiency
    - Condition: Pairwise geometric features (alignment + distance) + semantic features
    
    Args:
        cond_dim: Condition embedding dimension (default: 128)
        num_train_timesteps: Training diffusion steps (default: 1000)
        num_inference_steps: Inference DDIM steps (default: 20)
        All other args inherited from CGTopoHead
    """
    
    def __init__(self,
                 *args,
                 cond_dim=128,
                 num_train_timesteps=1000,
                 num_inference_steps=20,
                 beta_schedule='linear',
                 **kwargs):
        
        super(CGTopoHeadDiffusion, self).__init__(*args, **kwargs)
        
        # Remove original topology predictor (replaced by diffusion model)
        del self.vertex_inteact
        del self.lclc_branch
        
        # ========== Diffusion Components ==========
        self.cond_encoder = ConditionEncoder(
            query_dim=self.embed_dims,
            geo_dim=4,
            cond_dim=cond_dim
        )
        
        self.denoising_net = RelationTransformer(
            d_model=cond_dim,
            nhead=4,
            num_layers=3
        )
        
        self.diffusion_scheduler = GaussianDiffusion(
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            beta_schedule=beta_schedule
        )
        
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    def forward(self, mlvl_feats, lidar_feat, img_metas, prev_bev=None, only_bev=False):
        """
        Override forward to integrate diffusion-based topology prediction.
        
        Training Mode:
            - Computes condition features from query_feat and pts_preds
            - Returns condition for loss computation (no denoising loop)
        
        Inference Mode:
            - Runs full DDIM sampling loop (20 steps)
            - Returns denoised adjacency matrix
        """
        if only_bev:
            return super().forward(mlvl_feats, lidar_feat, img_metas, prev_bev, only_bev)
        
        # ========== Call Parent Forward (Get Decoder Outputs) ==========
        # Standard DETR processing up to topology prediction
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
        outputs_conds = []  # Store conditions for each layer
        
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
            # CRITICAL: .detach() prevents gradient backprop to Transformer Decoder
            # This ensures detection accuracy (loss_pts, loss_cls) is NOT affected by diffusion training
            vertex_feat = hs[lvl].view(bs, self.num_vec, self.num_pts_per_vec, -1).mean(2).detach()
            # [B, N, D] - detached from computation graph
            
            # Get predicted lane points (denormalize to physical space)
            # 1. 克隆 tmp 以免影响后续的原地操作
            tmp_for_cond = tmp.clone()
            
            # 2. 像检测头一样，加上参考点并 Sigmoid，得到 [0, 1] 的归一化坐标
            tmp_for_cond[..., 0:2] += reference[..., 0:2]
            tmp_for_cond = tmp_for_cond.sigmoid()
            
            # 3. 现在 tmp_for_cond 是正确的 [0, 1] 坐标，可以反归一化了
            tmp_pts_normalized = tmp_for_cond.view(bs, self.num_vec, self.num_pts_per_vec, 2)
            from .BitLine_head import denormalize_2d_pts
            # CRITICAL: .detach() prevents gradient backprop to regression branch
            # This protects geometric prediction (loss_pts, loss_dir) from diffusion training
            tmp_pts_physical = denormalize_2d_pts(
                tmp_pts_normalized.view(bs, -1, 2),
                self.pc_range
            ).view(bs, self.num_vec, self.num_pts_per_vec, 2).detach()
            # [B, N, P, 2] in physical space - detached from computation graph
            
            # ========== Compute Condition Features ==========
            cond = self.cond_encoder(vertex_feat, tmp_pts_physical)
            # [B, N, N, cond_dim]
            outputs_conds.append(cond)
            
            # ========== Topology Prediction ==========
            if self.training:
                # Training: Don't run denoising loop, just store condition
                # Actual adjacency will be predicted during loss computation
                out_sm = torch.zeros((bs, self.num_vec, self.num_vec),
                                    dtype=hs.dtype, device=hs.device)
            else:
                # Inference: Run DDIM sampling
                out_sm = self.ddim_sampling(cond)  # [B, N, N]
            
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
            'all_conds': outputs_conds if self.training else None,  # 因为计算 Loss 的步骤发生在 forward 之外,除了train不需要计算loss
        }

        return outs
    
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
        Override get_targets to also return pos_inds and pos_gt_inds for each sample.
        
        This avoids redundant assignment computation in loss_single.
        """
        # Call parent's get_targets
        targets = super().get_targets(
            cls_scores_list, bbox_preds_list, pts_preds_list, sms_preds_list,
            hs_preds_list, gt_bboxes_list, gt_labels_list, gt_shifts_pts_list,
            gt_sms_list, gt_control_pts_list, gt_bboxes_ignore_list
        )
        
        # Unpack parent's return values
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list, sms_targets_list, loss_beizer_list,
         num_total_pos, num_total_neg) = targets
        
        # Re-compute assignments to get pos_inds and pos_gt_inds for each sample
        pos_inds_list = []
        pos_gt_inds_list = []
        
        for i in range(len(cls_scores_list)):
            cls_score = cls_scores_list[i]
            bbox_pred = bbox_preds_list[i]
            pts_pred = pts_preds_list[i]
            gt_bboxes = gt_bboxes_list[i]
            gt_labels = gt_labels_list[i]
            gt_shifts_pts = gt_shifts_pts_list[i]
            
            assign_result, _ = self.assigner.assign(
                bbox_pred, cls_score, pts_pred,
                gt_bboxes, gt_labels, gt_shifts_pts,
                gt_bboxes_ignore_list[i] if gt_bboxes_ignore_list is not None else None
            )
            sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
            
            pos_inds_list.append(sampling_result.pos_inds)
            pos_gt_inds_list.append(sampling_result.pos_assigned_gt_inds)
        
        # Return extended tuple with pos_inds and pos_gt_inds
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                pts_targets_list, pts_weights_list, sms_targets_list, loss_beizer_list,
                num_total_pos, num_total_neg, pos_inds_list, pos_gt_inds_list)
    
    @torch.no_grad()
    def ddim_sampling(self, cond):
        """
        DDIM sampling for inference: Generate adjacency matrix from pure noise.
        
        Args:
            cond: [B, N, N, D] Condition features
            
        Returns:
            adj_pred: [B, N, N] Predicted adjacency matrix (0 or 1)
        """
        B, N, _, D = cond.shape
        device = cond.device
        
        # Start from pure Gaussian noise
        x_t = torch.randn(B, N, N, 1, device=device)  # [B, N, N, 1]
        
        # DDIM timestep schedule (reverse order: T -> 0)
        timesteps = torch.linspace(
            self.num_train_timesteps - 1, 0, self.num_inference_steps, dtype=torch.long, device=device
        )
        
        # Iterative denoising
        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Get next timestep
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
            else:
                t_prev = torch.tensor(0, device=device)
            
            t_prev_batch = torch.full((B,), t_prev, device=device, dtype=torch.long)
            
            # DDIM step
            x_t = self.diffusion_scheduler.ddim_step(
                model=self.denoising_net,
                x_t=x_t,
                t=t_batch,
                t_prev=t_prev_batch,
                cond=cond,
                eta=0.0  # Deterministic
            )
        
        # Final output: sigmoid + threshold
        adj_pred = torch.sigmoid(x_t.squeeze(-1))  # [B, N, N]
        adj_pred = (adj_pred > 0.5).float()  # Binarize
        
        return adj_pred
    
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
                    pos_inds_list,
                    pos_gt_inds_list,
                    gt_bboxes_ignore_list=None):
        """
        Override loss_single to integrate diffusion loss.
        
        Key Change:
            - Replaces BCE loss with diffusion training loss
            - Randomly samples timestep t, adds noise, predicts noise
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        sms_preds_list = [sms_preds[i] for i in range(num_imgs)]
        hs_preds_list = [hs_preds[i] for i in range(num_imgs)]
        cond_list = [cond[i] for i in range(num_imgs)]
        
        # Get targets (including positive sample indices)
        cls_reg_targets = self.get_targets(
            cls_scores_list, bbox_preds_list, pts_preds_list, sms_preds_list,
            hs_preds_list, gt_bboxes_list, gt_labels_list, gt_shifts_pts_list,
            gt_sms_list, gt_control_pts_list, gt_bboxes_ignore_list
        )
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list, _, loss_beizer_list,
         num_total_pos, num_total_neg, _, _) = cls_reg_targets
        
        # ========== Diffusion Loss ==========
        # Compute loss only on positive samples (matched queries)
        # Use pre-computed pos_inds and pos_gt_inds from get_targets (no redundant assignment)
        loss_adj_list = []
        
        for i in range(len(cond_list)):
            cond_i = cond_list[i]
            gt_sms = gt_sms_list[i]
            
            # Use pre-computed indices from get_targets
            pos_inds = pos_inds_list[i]  # Query indices of matched samples
            pos_gt_inds = pos_gt_inds_list[i]  # Corresponding GT indices
            
            if len(pos_inds) == 0:
                loss_adj_list.append(torch.tensor(0.0, device=cond_i.device))
                continue
            
            # Extract positive subgraph using CORRECT indices
            # cond_pos uses query indices (pos_inds)
            # gt_adj_pos uses GT indices (pos_gt_inds)
            cond_pos = cond_i[pos_inds][:, pos_inds]  # [P, P, D]
            gt_adj_pos = gt_sms[pos_gt_inds][:, pos_gt_inds].float().unsqueeze(-1)  # [P, P, 1] 
            
            # Random timestep
            P = len(pos_inds)
            t = torch.randint(0, self.num_train_timesteps, (1,), device=cond_i.device).expand(P)
            
            # Forward diffusion: add noise
            noise = torch.randn_like(gt_adj_pos)
            noisy_adj = self.diffusion_scheduler.q_sample(gt_adj_pos.unsqueeze(0), t[:1], noise=noise.unsqueeze(0))
            # [1, P, P, 1]
            
            # Predict noise
            noise_pred = self.denoising_net(noisy_adj, t[:1], cond_pos.unsqueeze(0))
            # [1, P, P, 1]
            
            # MSE loss
            loss_adj = F.mse_loss(noise_pred, noise.unsqueeze(0))
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
        
        from mmcv.utils import TORCH_VERSION, digit_version
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)
            loss_adj = torch.nan_to_num(loss_adj)
        
        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir, loss_adj, loss_beizer
    
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
        all_conds = preds_dicts['all_conds']  # NEW: condition features
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
        
        # Pre-compute pos_inds and pos_gt_inds once for all layers (optimization)
        # We compute this from the first layer's predictions
        _, _, _, _, _, _, _, _, _, _, pos_inds_list, pos_gt_inds_list = self.get_targets(
            [all_cls_scores[0][i] for i in range(len(gt_bboxes_list))],
            [all_bbox_preds[0][i] for i in range(len(gt_bboxes_list))],
            [all_pts_preds[0][i] for i in range(len(gt_bboxes_list))],
            [all_sms_preds[0][i] for i in range(len(gt_bboxes_list))],
            [all_hs_preds[0][i] for i in range(len(gt_bboxes_list))],
            gt_bboxes_list, gt_labels_list, gt_shifts_pts_list,
            gt_sms_list, gt_control_pts_list, gt_bboxes_ignore
        )
        
        # Replicate for all layers (pos_inds computed from first layer can be used for all)
        all_pos_inds_list = [pos_inds_list for _ in range(num_dec_layers)]
        all_pos_gt_inds_list = [pos_gt_inds_list for _ in range(num_dec_layers)]

        # Compute losses for each layer
        from mmdet.core import multi_apply
        losses_cls, losses_bbox, losses_iou, losses_pts, losses_dir, losses_adj, losses_beizer = multi_apply(
            self.loss_single,
            all_cls_scores, all_bbox_preds, all_pts_preds, all_sms_preds,
            all_hs_preds, all_conds,  # Pass conditions
            all_gt_bboxes_list, all_gt_labels_list, all_gt_shifts_pts_list,
            all_gt_sms_list, all_gt_control_pts_list,
            all_pos_inds_list, all_pos_gt_inds_list,  # Pass pre-computed indices
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

        # Intermediate layer losses
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i, loss_dir_i, loss_adj_i, loss_beizer_i in zip(
                losses_cls[:-1], losses_pts[:-1], losses_dir[:-1],
                losses_adj[:-1], losses_beizer[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
            loss_dict[f'd{num_dec_layer}.loss_adj'] = loss_adj_i
            loss_dict[f'd{num_dec_layer}.loss_beizer'] = loss_beizer_i
            num_dec_layer += 1
        
        return loss_dict, None, None
