"""
MapTR Decoder with DAM (Decoder cross-Attention Map) Analysis Support

这个文件实现了支持DAM分析的MapTR解码器。
DAM分析用于计算解码器的对象查询引用了多少编码器输出标记（非零注意力权重的比例）。

参考: Sparse DETR - "Efficient DETR: Improving End-to-End Object Detector with Dense Prior"
"""

import torch
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmdet.models.utils.transformer import inverse_sigmoid


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapTRDecoderWithDAM(TransformerLayerSequence):
    """
    实现支持DAM分析的DETR3D transformer解码器
    
    DAM (Decoder cross-Attention Map) 分析:
    - 统计解码器对象查询引用的编码器标记数量
    - 计算非零注意力权重的编码器token比例
    - 帮助理解模型的稀疏性和效率
    
    Args:
        return_intermediate (bool): 是否返回中间层输出
        enable_dam_analysis (bool): 是否启用DAM分析（仅在推理时使用）
    """

    def __init__(self, *args, return_intermediate=False, enable_dam_analysis=False, **kwargs):
        super(MapTRDecoderWithDAM, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False
        
        # DAM分析相关参数
        self.enable_dam_analysis = enable_dam_analysis
        self.dam_statistics = {
            'total_encoder_tokens': 0,      # 编码器token总数
            'referenced_encoder_tokens': 0,  # 被引用的编码器token数
            'total_samples': 0,              # 样本总数
            'layer_statistics': [],          # 每层的统计信息
        }

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):
        """
        前向传播函数，支持DAM分析
        
        Args:
            query (Tensor): 输入查询，形状 `(num_query, bs, embed_dims)`
            reference_points (Tensor): 参考点，形状 (bs, num_query, 2)
            reg_branch: 用于细化回归结果的模块列表
            
        Returns:
            Tensor: 结果，形状 [num_layers, num_query, bs, embed_dims]（当return_intermediate为True）
                   或 [1, num_query, bs, embed_dims]
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        
        # 如果启用DAM分析，初始化层级统计
        if self.enable_dam_analysis:
            layer_dam_stats = []
        
        for lid, layer in enumerate(self.layers):
            # 准备当前层的参考点输入
            reference_points_input = reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
            
            # 调用transformer层的前向传播
            # 注意：我们需要从层中获取注意力权重用于DAM分析
            if self.enable_dam_analysis:
                # 在DAM分析模式下，需要从层中提取注意力权重
                output = layer(
                    output,
                    *args,
                    reference_points=reference_points_input,
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                
                # 尝试从层中获取最后一次计算的注意力权重
                # 注意：这需要层的注意力模块支持返回注意力权重
                if hasattr(layer, 'attentions') and len(layer.attentions) > 0:
                    # 获取cross attention (通常是第二个注意力模块)
                    cross_attn_module = layer.attentions[1] if len(layer.attentions) > 1 else None
                    if cross_attn_module is not None and hasattr(cross_attn_module, 'last_attn_weights'):
                        attn_weights = cross_attn_module.last_attn_weights
                        # 计算当前层的DAM统计
                        layer_stat = self._compute_dam_for_layer(attn_weights)
                        layer_dam_stats.append(layer_stat)
            else:
                # 正常前向传播（训练模式）
                output = layer(
                    output,
                    *args,
                    reference_points=reference_points_input,
                    key_padding_mask=key_padding_mask,
                    **kwargs)
            
            # 调整输出形状
            output = output.permute(1, 0, 2)

            # 如果有回归分支，更新参考点
            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])

                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            # 恢复输出形状
            output = output.permute(1, 0, 2)
            
            # 保存中间结果
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        
        # 如果启用DAM分析，保存层级统计
        if self.enable_dam_analysis and len(layer_dam_stats) > 0:
            self.dam_statistics['layer_statistics'] = layer_dam_stats

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points

    def _compute_dam_for_layer(self, attn_weights):
        """
        计算单层的DAM统计信息
        
        Args:
            attn_weights: 可以是字典（deformable attention）或张量
        
        Returns:
            dict: 包含该层DAM统计的字典，或None（留待后续分析）
        """
        # 对于Deformable Attention，我们只保存数据，不在这里计算
        # 实际计算在DAMAnalyzer中进行
        return None  # 暂时返回None，由外部分析器处理
        return {
            'mean_reference_ratio': reference_ratio.mean().item(),
            'min_reference_ratio': reference_ratio.min().item(),
            'max_reference_ratio': reference_ratio.max().item(),
            'total_encoder_tokens': total_tokens.item(),
            'mean_referenced_tokens': referenced_tokens.mean().item(),
        }

    def get_dam_statistics(self):
        """
        获取DAM统计信息
        
        Returns:
            dict: 包含完整DAM分析结果的字典
        """
        return self.dam_statistics
    
    def reset_dam_statistics(self):
        """重置DAM统计信息"""
        self.dam_statistics = {
            'total_encoder_tokens': 0,
            'referenced_encoder_tokens': 0,
            'total_samples': 0,
            'layer_statistics': [],
        }
    
    def enable_dam_analysis_mode(self):
        """启用DAM分析模式"""
        self.enable_dam_analysis = True
        self.reset_dam_statistics()
        print("DAM Analysis mode enabled")
    
    def disable_dam_analysis_mode(self):
        """禁用DAM分析模式"""
        self.enable_dam_analysis = False
        print("DAM Analysis mode disabled")
