# Copyright (C) 2024 Xiaomi Corporation.
# Sparse DETR Implementation - Sparse Transformer
# 集成DAM预测器和稀疏化BEV Encoder (支持BEVFormer)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.runner.base_module import BaseModule
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from torch.nn.init import normal_
from mmdet.models.utils.builder import TRANSFORMER
from .transformer import JAPerceptionTransformer, KPALayer
from .encoder_sparse import gather_tokens_by_indices, scatter_tokens_by_indices
from .mask_predictor_sparse import MaskPredictor


@TRANSFORMER.register_module()
class JAPerceptionTransformerSparse(JAPerceptionTransformer):
    """
    稀疏化的BEV Transformer - Sparse DETR实现 (支持BEVFormer)
    
    核心改动:
        1. encoder支持稀疏化(rho参数控制)
        2. 集成DAM预测器用于token选择  
        3. 实现gen_encoder_output_proposals生成backbone proposals
        4. 在get_bev_features中返回mask预测信息
        5. decoder保持完整,可以访问所有BEV tokens
        
    稀疏化流程:
        - Encoder阶段: 只更新rho比例的重要tokens (例如10%)
        - Decoder阶段: 完整的cross-attention,访问所有BEV特征
        
    新增参数:
        rho: 稀疏率 (0.1表示保留10%的tokens)
        mask_predictor_dim: DAM预测器隐藏层维度
    """
    
    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 fuser=None,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 modality='vision',
                 rho=0.5,  # 保留比例 (0.5表示保留50%的tokens)
                 mask_predictor_dim=256,  # DAM预测器维度
                 use_enc_aux_loss=False,  # 【新增】是否启用encoder辅助损失
                 sparse_encoder=True,  # 是否启用稀疏encoder (已废弃,通过rho控制)
                 **kwargs):
        super(JAPerceptionTransformerSparse, self).__init__(
            num_feature_levels=num_feature_levels,
            num_cams=num_cams,
            two_stage_num_proposals=two_stage_num_proposals,
            fuser=fuser,
            encoder=encoder,
            decoder=decoder,
            embed_dims=embed_dims,
            rotate_prev_bev=rotate_prev_bev,
            use_shift=use_shift,
            use_can_bus=use_can_bus,
            can_bus_norm=can_bus_norm,
            use_cams_embeds=use_cams_embeds,
            rotate_center=rotate_center,
            modality=modality,
            **kwargs
        )
        
        # 稀疏化参数
        self.rho = rho
        self.sparse_enabled = rho > 0
        self.use_enc_aux_loss = use_enc_aux_loss  # 【新增】Encoder辅助损失标志
        
        # 【方案2: 独立RNG流】
        # 保存全局RNG状态,用于创建sparse模块时隔离
        # 这样sparse模块不会影响共享模块的初始化
        global_rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            global_cuda_rng_states = [torch.cuda.get_rng_state(i) 
                                      for i in range(torch.cuda.device_count())]
        
        # DAM预测器 - 用于预测token重要性 (仅BEVFormer需要)
        if self.sparse_enabled and self.use_attn_bev:
            # 使用独立的RNG seed创建sparse模块
            # 这样不会影响后续共享模块的初始化
            sparse_seed = 42  # 与主seed不同,避免冲突
            
            # 临时切换到独立RNG状态
            torch.manual_seed(sparse_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sparse_seed)
            
            self.mask_predictor = MaskPredictor(
                in_dim=embed_dims,
                h_dim=mask_predictor_dim
            )
            # Encoder output处理层 (对应Sparse DETR的enc_output)
            self.enc_output = nn.Linear(embed_dims, embed_dims)
            self.enc_output_norm = nn.LayerNorm(embed_dims)
        else:
            self.enc_output = None
            self.enc_output_norm = None
        
        # 【Encoder辅助损失】用于encoder中间层的预测头
        # 参考Sparse DETR: 如果启用use_enc_aux_loss,需要为encoder每一层配置检测头
        # 这些头会在Head中被设置 (类似Sparse DETR的transformer.encoder.aux_heads)
        if self.use_enc_aux_loss and self.sparse_enabled:
            # encoder.aux_heads将在head的__init__中设置为True
            # encoder.class_embed和encoder.bbox_embed也会在head中设置
            pass
            
        # 【关键】恢复全局RNG状态
        # 这样后续的模块初始化不受sparse模块影响
        torch.set_rng_state(global_rng_state)
        if torch.cuda.is_available():
                for i, state in enumerate(global_cuda_rng_states):
                    torch.cuda.set_rng_state(state, device=i)
        else:
            self.mask_predictor = None
        
        # 用于存储mask预测结果(供loss计算使用)
        self.backbone_mask_prediction = None
        self.backbone_topk_proposals = None
        
        # Warmup冻结标记
        self._mask_predictor_frozen = False
    
    def freeze_mask_predictor(self):
        """冻结MaskPredictor相关参数(warmup期间)"""
        if self.mask_predictor is not None and not self._mask_predictor_frozen:
            for param in self.mask_predictor.parameters():
                param.requires_grad = False
            for param in self.enc_output.parameters():
                param.requires_grad = False
            for param in self.enc_output_norm.parameters():
                param.requires_grad = False
            self._mask_predictor_frozen = True
    
    def unfreeze_mask_predictor(self):
        """解冻MaskPredictor相关参数(warmup结束后)"""
        if self.mask_predictor is not None and self._mask_predictor_frozen:
            for param in self.mask_predictor.parameters():
                param.requires_grad = True
            for param in self.enc_output.parameters():
                param.requires_grad = True
            for param in self.enc_output_norm.parameters():
                param.requires_grad = True
            self._mask_predictor_frozen = False
    
    def init_weights(self):
        """
        初始化Sparse Transformer权重
        
        【方案2: 独立RNG流】
        - 父类初始化使用全局RNG (与原版一致)
        - Sparse模块初始化使用独立RNG (不影响全局状态)
        """
        # 1. 保存当前全局RNG状态
        global_rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            global_cuda_rng_states = [torch.cuda.get_rng_state(i) 
                                      for i in range(torch.cuda.device_count())]
        
        # 2. 调用父类的初始化方法(原版的标准初始化)
        #    这会使用全局RNG,与原版完全相同
        super().init_weights()
        
        # 3. 保存父类初始化后的RNG状态
        after_shared_rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            after_shared_cuda_rng_states = [torch.cuda.get_rng_state(i) 
                                            for i in range(torch.cuda.device_count())]
        
        # 4. 为Sparse特有的模块使用独立的RNG流
        if self.mask_predictor is not None:
            # 切换到独立RNG (与__init__中相同的seed)
            sparse_seed = 1234 + 999
            torch.manual_seed(sparse_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sparse_seed)
            
            # 使用Xavier uniform初始化 (与父类一致的方法)
            for m in self.mask_predictor.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            
            nn.init.xavier_uniform_(self.enc_output.weight)
            if self.enc_output.bias is not None:
                nn.init.constant_(self.enc_output.bias, 0)
            
            # 5. 恢复父类初始化后的RNG状态
            #    确保后续任何操作不受sparse初始化影响
            torch.set_rng_state(after_shared_rng_state)
            if torch.cuda.is_available():
                for i, state in enumerate(after_shared_cuda_rng_states):
                    torch.cuda.set_rng_state(state, device=i)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shape):
        """
        生成encoder输出proposals
        
        对应Sparse DETR中的gen_encoder_output_proposals函数
        参考: /home/ubuntu/disk4/jmk3/Project/sparse-detr/models/deformable_transformer.py
        
        Args:
            memory: [B, H*W, C] BEV特征 (例如 [B, 20000, 256])
            memory_padding_mask: [B, H*W] padding mask (BEV通常没有padding,全False)
            spatial_shape: (H, W) BEV空间尺寸,例如 (200, 100)
            
        Returns:
            output_memory: [B, H*W, C] 归一化后的memory
            output_proposals: [B, H*W, 2] 归一化的位置坐标 (x, y)
            valid_token_nums: [B] 每个batch的有效token数量
            
        功能:
            1. 为每个BEV位置生成归一化坐标作为proposal
            2. 处理memory (加LayerNorm)
            3. 计算有效token数量(排除padding)
            
        注意:
            - 与Sparse DETR不同,BEV特征是单尺度的,不需要multi-scale处理
            - BEV通常没有padding,所以valid_token_nums就是H*W
        """
        B, L, C = memory.shape
        H, W = spatial_shape
        
        # 生成归一化的网格坐标 [0, 1]
        # 对应Sparse DETR中的output_proposals
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=memory.dtype, device=memory.device),
            torch.linspace(0.5, W - 0.5, W, dtype=memory.dtype, device=memory.device)
        )
        grid_y = grid_y.reshape(-1) / H  # [H*W] 归一化到[0,1]
        grid_x = grid_x.reshape(-1) / W  # [H*W]
        
        # 堆叠为proposals: [H*W, 2]
        output_proposals = torch.stack([grid_x, grid_y], -1)
        output_proposals = output_proposals.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, 2]
        
        # 处理memory: 类似Sparse DETR的enc_output + norm
        # 注意: 这里不使用mask过滤,因为BEV特征通常没有padding
        output_memory = self.enc_output_norm(self.enc_output(memory))
        
        # 计算有效token数量
        # BEV特征通常没有padding,所以都是有效的
        if memory_padding_mask is not None:
            valid_token_nums = (~memory_padding_mask).sum(dim=1)  # [B]
        else:
            valid_token_nums = torch.full((B,), L, dtype=torch.long, device=memory.device)
        
        return output_memory, output_proposals, valid_token_nums

    def get_bev_features_sparse(self,
                                mlvl_feats,
                                lidar_feat,
                                bev_queries,
                                bev_h,
                                bev_w,
                                grid_length=[0.512, 0.512],
                                bev_pos=None,
                                prev_bev=None,
                                return_mask_pred=False,
                                cls_branches=None,
                                reg_branches=None,
                                **kwargs):
        """
        获取BEV特征 - 支持稀疏化 (BEVFormer + LSS均支持)
        
        Args:
            return_mask_pred: 是否返回mask预测(训练时为True)
            其他参数同父类
            
        Returns:
            bev_embed: BEV特征 [num_query, bs, C] 或 [bs, H*W, C]
            
        稀疏化流程:
            【BEVFormer路径】:
            1. 准备BEV queries + position encoding
            2. 调用gen_encoder_output_proposals生成proposals
            3. MaskPredictor预测token重要性
            4. Top-K选择
            5. Encoder处理(通过topk_indices参数启用稀疏更新)
            
            【LSS路径】:
            1. encoder前向传播时,如果使用LSSTransformSparse,
               会自动进行DAM预测和Top-K选择
            2. 保存mask预测结果供后续loss计算
        """
        # 检查encoder类型
        if self.use_attn_bev:
            # 【BEVFormer路径】- 支持稀疏化
            bev_embed = self.attn_bev_encode(
                mlvl_feats,
                bev_queries,
                bev_h,
                bev_w,
                grid_length=grid_length,
                bev_pos=bev_pos,
                prev_bev=prev_bev,
                return_mask_pred=return_mask_pred,  # 传递参数
                cls_branches=cls_branches,  # 传递分类分支
                reg_branches=reg_branches,  # 传递回归分支
                **kwargs)
            return bev_embed
        else:
            # 【LSS路径】- 支持稀疏化
            # 调用encoder的forward,如果是LSSTransformSparse会返回mask预测
            if hasattr(self.encoder, 'enable_sparse') and self.encoder.enable_sparse and return_mask_pred:
                # 稀疏化模式: 返回完整特征 + mask预测
                # 提取图像 (LSS只支持单层特征)
                assert len(mlvl_feats) == 1, 'Currently we only support single level feat in LSS'
                images = mlvl_feats[0]
                
                bev_embed, backbone_mask_prediction, backbone_topk_proposals = \
                    self.encoder(images, img_metas=kwargs.get('img_metas'), return_mask_pred=True)
                
                # 保存mask预测结果
                self.backbone_mask_prediction = backbone_mask_prediction
                self.backbone_topk_proposals = backbone_topk_proposals
            else:
                # 正常模式: 只返回特征
                bev_embed = self.lss_bev_encode(mlvl_feats, prev_bev=prev_bev, **kwargs)
                self.backbone_mask_prediction = None
                self.backbone_topk_proposals = None
            
            # 融合lidar特征(如果有)
            if lidar_feat is not None:
                bs = mlvl_feats[0].size(0)
                bev_embed_reshape = bev_embed.view(bs, bev_h, bev_w, -1).permute(0, 3, 1, 2).contiguous()
                lidar_feat = lidar_feat.permute(0, 1, 3, 2).contiguous()
                lidar_feat = nn.functional.interpolate(lidar_feat, size=(bev_h, bev_w), 
                                                      mode='bicubic', align_corners=False)
                fused_bev = self.fuser([bev_embed_reshape, lidar_feat])
                bev_embed = fused_bev.flatten(2).permute(0, 2, 1).contiguous()

            return bev_embed

    def attn_bev_encode(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            return_mask_pred=False,  # 新增: 控制是否返回mask预测
            cls_branches=None,  # 新增: 分类分支 (用于encoder aux loss)
            reg_branches=None,  # 新增: 回归分支 (用于encoder aux loss)
            **kwargs):
        """
        BEV Attention编码 - 支持Sparse DETR稀疏化
        
        新增参数:
            return_mask_pred: 训练时设为True,用于DAM预测和Top-K选择
            
        Sparse DETR流程:
            1. 准备BEV queries + ego motion处理
            2. 如果启用稀疏化:
               a) 使用gen_encoder_output_proposals生成proposals
               b) MaskPredictor预测token重要性
               c) Top-K选择重要tokens
            3. 调用encoder(传入topk_indices启用稀疏更新)
            4. 返回完整BEV特征(仅部分被更新)
        """
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # 步骤1: ego motion处理 (与父类相同)
        from torchvision.transforms.functional import rotate
        delta_x = np.array([each['can_bus'][0] for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1] for each in kwargs['img_metas']])
        ego_angle = np.array([each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        # 优化: 先转换为numpy数组再创建tensor，避免警告
        shift = bev_queries.new_tensor(np.stack([shift_x, shift_y], axis=1))

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # 步骤2: 添加CAN bus信息
        # 优化: 先转换为numpy数组再创建tensor，避免警告
        can_bus_list = [each['can_bus'] for each in kwargs['img_metas']]
        can_bus = bev_queries.new_tensor(np.array(can_bus_list))
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        # 步骤3: 准备多尺度图像特征
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        # 步骤4: Sparse DETR核心 - DAM预测和Top-K选择
        topk_indices = None
        if self.sparse_enabled and return_mask_pred:
            # 4.1 准备encoder input memory (bev_queries + bev_pos)
            # 转换为 [bs, H*W, C] 格式
            bev_queries_for_pred = bev_queries.permute(1, 0, 2)  # [bs, H*W, C]
            bev_pos_for_pred = bev_pos.permute(1, 0, 2)  # [bs, H*W, C]
            backbone_output_memory = bev_queries_for_pred + bev_pos_for_pred
            
            # 4.2 生成encoder output proposals
            output_memory, output_proposals, valid_token_nums = self.gen_encoder_output_proposals(
                backbone_output_memory, 
                memory_padding_mask=None,  # BEV没有padding
                spatial_shape=(bev_h, bev_w)
            )
            
            # 4.3 MaskPredictor预测token重要性
            backbone_mask_prediction = self.mask_predictor(output_memory).squeeze(-1)  # [B, H*W] torch.Size([4, 20000])
            
            # 4.4 Top-K选择 (保持空间顺序)
            sparse_token_nums = (valid_token_nums.float() * self.rho).int()  # [B] tensor([2001, 2001, 2001, 2001]去掉了 + 1 
            backbone_topk = int(max(sparse_token_nums))  # 取最大值作为K
            backbone_topk = min(backbone_topk, backbone_mask_prediction.shape[1])
            
            # 【关键修复】选择top-k个最重要的tokens,并保持空间顺序
            # 问题: torch.topk会按score降序重排索引,破坏BEV的空间结构
            # 解决: topk后sort,始终恢复空间顺序 (无论rho=1.0还是<1.0)
            topk_unsorted = torch.topk(backbone_mask_prediction, backbone_topk, dim=1)[1]  # [B, K]
            backbone_topk_proposals = torch.sort(topk_unsorted, dim=1)[0]  # 恢复空间顺序
            
            # 保存用于loss计算
            self.backbone_mask_prediction = backbone_mask_prediction
            self.backbone_topk_proposals = backbone_topk_proposals
            topk_indices = backbone_topk_proposals
            
            # 对应Sparse DETR的output_proposals (用于encoder aux loss,如果需要)
            # 这里传递完整的proposals,encoder会根据topk_indices进行gather
            encoder_output_proposals = output_proposals  # [B, H*W, 2]
            encoder_sparse_token_nums = sparse_token_nums  # [B]
        else:
            self.backbone_mask_prediction = None
            self.backbone_topk_proposals = None
            encoder_output_proposals = None
            encoder_sparse_token_nums = None

        # 步骤5: 调用encoder (启用稀疏化)
        # 如果topk_indices不为None且encoder支持,则传入启用Sparse DETR
        # 完整传递三个参数: topk_indices, output_proposals, sparse_token_nums
        # 【新增】传递cls_branches和reg_branches用于encoder辅助损失
        encoder_output = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            topk_indices=topk_indices,  # [B, K] top-k索引
            output_proposals=encoder_output_proposals,  # [B, H*W, 2] 位置proposals (用于aux loss)
            sparse_token_nums=encoder_sparse_token_nums,  # [B] 每个样本的稀疏token数量
            cls_branches=cls_branches,  # 分类分支 (用于encoder aux loss)
            reg_branches=reg_branches,  # 回归分支 (用于encoder aux loss)
            **kwargs
        )
        
        # 处理encoder返回值
        # 【Encoder辅助损失 - 语义分割版本】
        # 如果启用了encoder辅助损失,encoder会返回(bev_embed, enc_inter_class)
        # 否则只返回bev_embed
        if isinstance(encoder_output, tuple) and len(encoder_output) == 2:
            bev_embed, enc_inter_outputs_class = encoder_output
            # 保存encoder中间层输出 (只有分类,没有坐标)
            self.enc_inter_outputs_class = enc_inter_outputs_class
        else:
            bev_embed = encoder_output
            self.enc_inter_outputs_class = None
        
        # encoder返回[len_bev, bs, C],需要permute为[bs, len_bev, C]与原版一致
        bev_embed = bev_embed.permute(1, 0, 2)
        
        return bev_embed

    def forward(self,
                mlvl_feats,
                lidar_feat,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                return_mask_pred=False,  # 新增: 控制是否返回mask预测
                **kwargs):
        """
        前向传播 - 支持稀疏化
        
        新增参数:
            return_mask_pred: 训练时设为True,返回mask预测供损失计算
            
        返回值:
            同父类,如果return_mask_pred=True,会在最后额外返回:
                - backbone_mask_prediction: [B, H*W]
                - backbone_topk_proposals: [B, K]
        """
        # 步骤1: 获取BEV特征(可能包含稀疏化)
        bev_embed = self.get_bev_features_sparse(
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            return_mask_pred=return_mask_pred,
            cls_branches=cls_branches,  # 传递分类分支
            reg_branches=reg_branches,  # 传递回归分支
            **kwargs
        )  # bev_embed: [B, H*W, C]

        # 步骤2: 准备decoder queries
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        
        # 步骤3: 生成reference points
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        # 步骤4: BEV关键点增强(保持原有逻辑)
        # bev_embed现在是[bs, len_bev, C]格式,与原版一致
        kp_bev_embed = bev_embed.permute(0, 2, 1).reshape(bs, -1, bev_h, bev_w).contiguous()
        kp_bev_embed = self.bev_keypoint_decoder(kp_bev_embed).contiguous()
        kp_bev_preds = self.bev_keypoint_proj(kp_bev_embed).contiguous()
        kp_bev_embed_flatten = self.maxpool(kp_bev_embed).reshape(bs, self.embed_dims, -1).contiguous()
        query_pos = self.query_enhance(query_pos, kp_bev_embed_flatten.permute(0, 2, 1)).contiguous()

        # 步骤5: Decoder迭代细化
        # 注意: decoder使用完整的bev_embed,可以访问所有位置
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        # 与原版一致,需要permute为[len_bev, bs, C]
        bev_embed = bev_embed.permute(1, 0, 2)

        # 调用decoder - 可能返回sampling locations和attention weights
        decoder_outputs = self.decoder(
            query=query,
            key=None,
            value=bev_embed,  # 完整的BEV特征
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs
        )
        
        # 解包decoder输出
        if len(decoder_outputs) == 4:
            # MapTRDecoderSparse返回: (inter_states, inter_references, sampling_locs, attn_weights)
            inter_states, inter_references, sampling_locations_dec, attn_weights_dec = decoder_outputs
        else:
            # 原始MapTRDecoder返回: (inter_states, inter_references)
            inter_states, inter_references = decoder_outputs
            sampling_locations_dec = None
            attn_weights_dec = None

        inter_references_out = inter_references
        
        # 保存spatial信息
        spatial_shapes = torch.tensor([[bev_h, bev_w]], device=query.device)
        level_start_index = torch.tensor([0], device=query.device)

        # 步骤6: 返回结果
        if return_mask_pred and self.backbone_mask_prediction is not None:
            # 训练模式: 返回mask预测、attention信息和encoder中间层输出
            # 检查是否有encoder中间层输出
            if hasattr(self, 'enc_inter_outputs_class') and self.enc_inter_outputs_class is not None:
                # 【语义分割版本】包含encoder辅助损失输出 (12个元素,只有分类)
                return (bev_embed, inter_states, init_reference_out, inter_references_out, 
                        kp_bev_preds, self.backbone_mask_prediction, self.backbone_topk_proposals,
                        sampling_locations_dec, attn_weights_dec, spatial_shapes, level_start_index,
                        self.enc_inter_outputs_class)
            else:
                # 不包含encoder辅助损失输出 (11个元素)
                return (bev_embed, inter_states, init_reference_out, inter_references_out, 
                        kp_bev_preds, self.backbone_mask_prediction, self.backbone_topk_proposals,
                        sampling_locations_dec, attn_weights_dec, spatial_shapes, level_start_index)
        else:
            # 推理模式: 正常返回
            return (bev_embed, inter_states, init_reference_out, inter_references_out, kp_bev_preds)

    def get_mask_prediction_info(self):
        """
        获取最近一次的mask预测信息
        用于计算DAM损失
        
        Returns:
            backbone_mask_prediction: [B, H*W] token重要性分数
            backbone_topk_proposals: [B, K] top-k indices
        """
        return self.backbone_mask_prediction, self.backbone_topk_proposals
