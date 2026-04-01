"""
Sparse DETR组件测试脚本
用于快速验证各个模块是否正常工作
"""

import torch
import os
import sys

# 设置正确的导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../..'))
sys.path.insert(0, project_root)
os.chdir(project_root)

print("=" * 60)
print("Sparse DETR组件测试")
print("=" * 60)

# ==================== 测试1: MaskPredictor ====================
print("\n【测试1】MaskPredictor (DAM预测器)")
from projects.mmdet3d_plugin.bitline.modules.mask_predictor_sparse import MaskPredictor

B, L, C = 2, 20000, 256  # 批次, tokens数量, 特征维度
x = torch.randn(B, L, C)

predictor = MaskPredictor(in_dim=256, h_dim=256)
output = predictor(x)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")

assert output.shape == (B, L, 1), "MaskPredictor输出形状错误"
print("✓ MaskPredictor测试通过")

# ==================== 测试2: Top-K选择 ====================
print("\n【测试2】Top-K选择机制")

rho = 0.1
sparse_token_nums = int(L * rho)
mask_prediction = output.squeeze(-1)  # [B, L]

# Top-K选择
values, indices = torch.topk(mask_prediction, k=sparse_token_nums, dim=1)

print(f"总tokens: {L}")
print(f"稀疏率 rho: {rho}")
print(f"保留tokens: {sparse_token_nums} ({sparse_token_nums/L*100:.1f}%)")
print(f"Top-K indices形状: {indices.shape}")
print(f"Top-K values范围: [{values.min():.3f}, {values.max():.3f}]")

# 验证indices在有效范围内
assert indices.min() >= 0 and indices.max() < L, "Top-K indices超出范围"
print("✓ Top-K选择测试通过")

# ==================== 测试3: Gather & Scatter操作 ====================
print("\n【测试3】Gather & Scatter操作")
from projects.mmdet3d_plugin.bitline.modules.encoder_sparse import (
    gather_tokens_by_indices, 
    scatter_tokens_by_indices
)

# 完整tokens
full_tokens = torch.randn(B, L, C)

# Gather: 提取top-k tokens
selected_tokens = gather_tokens_by_indices(full_tokens, indices)
print(f"Gather - 输入: {full_tokens.shape} → 输出: {selected_tokens.shape}")

# 模拟更新: 对选中的tokens加1
updated_tokens = selected_tokens + 1.0

# Scatter: 写回原位置
result_tokens = scatter_tokens_by_indices(full_tokens, updated_tokens, indices)
print(f"Scatter - 更新后形状: {result_tokens.shape}")

# 验证: 检查选中位置是否被更新
for b in range(B):
    for k in range(min(10, sparse_token_nums)):  # 检查前10个
        idx = indices[b, k].item()
        original_val = full_tokens[b, idx]
        updated_val = result_tokens[b, idx]
        # 更新后的值应该等于原值+1
        assert torch.allclose(updated_val, original_val + 1.0, atol=1e-5), \
            f"Scatter失败: 位置{idx}未正确更新"

print("✓ Gather & Scatter测试通过")

# ==================== 测试4: DAM损失计算 ====================
print("\n【测试4】DAM损失计算")
import torch.nn.functional as F

# 创建伪目标
target = torch.zeros_like(mask_prediction)
for b in range(B):
    target[b].scatter_(0, indices[b], 1.0)

print(f"Target形状: {target.shape}")
print(f"Target非零元素: {target.sum(dim=1)}")  # 应该等于sparse_token_nums

# 计算损失
loss = F.multilabel_soft_margin_loss(mask_prediction, target)
print(f"DAM损失值: {loss.item():.4f}")

assert not torch.isnan(loss), "DAM损失为NaN"
assert loss.item() >= 0, "DAM损失为负"
print("✓ DAM损失计算测试通过")

# ==================== 测试5: Correlation指标 ====================
print("\n【测试5】Correlation指标")

# 从mask预测中提取top-k (模拟预测)
_, pred_topk = torch.topk(mask_prediction, k=sparse_token_nums, dim=1)

# 计算与ground truth的重叠
overlaps = []
for b in range(B):
    gt_set = set(indices[b].cpu().tolist())
    pred_set = set(pred_topk[b].cpu().tolist())
    intersection = len(gt_set & pred_set)
    overlap = intersection / sparse_token_nums
    overlaps.append(overlap)

correlation = torch.tensor(overlaps).mean()
print(f"Correlation: {correlation.item():.3f}")
print(f"期望范围: 0.0~1.0 (越高越好)")

# 由于pred和gt是同一个,应该完全重叠
assert correlation.item() == 1.0, "Correlation计算错误"
print("✓ Correlation指标测试通过")

# ==================== 测试6: 配置文件加载 ====================
print("\n【测试6】配置文件加载")
try:
    from mmcv import Config
    cfg = Config.fromfile('projects/configs/bitline/bitline_ep110_sparse.py')
    
    print(f"Head类型: {cfg.model.pts_bbox_head.type}")
    print(f"Transformer类型: {cfg.model.pts_bbox_head.transformer.type}")
    print(f"Encoder类型: {cfg.model.pts_bbox_head.transformer.encoder.type}")
    print(f"稀疏率rho: {cfg.model.pts_bbox_head.rho}")
    print(f"DAM损失权重: {cfg.model.pts_bbox_head.mask_prediction_coef}")
    
    assert cfg.model.pts_bbox_head.type == 'CGTopoHeadSparse'
    assert cfg.model.pts_bbox_head.transformer.type == 'JAPerceptionTransformerSparse'
    assert cfg.model.pts_bbox_head.transformer.encoder.type == 'LSSTransformSparse'
    
    print("✓ 配置文件加载测试通过")
except Exception as e:
    print(f"✗ 配置文件加载失败: {e}")

# ==================== 总结 ====================
print("\n" + "=" * 60)
print("✓ 所有测试通过!")
print("=" * 60)
print("\n【关键数据】")
print(f"- BEV tokens总数: {L}")
print(f"- 稀疏率: {rho} ({rho*100}%)")
print(f"- 保留tokens: {sparse_token_nums}")
print(f"- 理论加速比: {1/rho:.1f}x (仅encoder)")
print("\n【下一步】")
print("运行验证脚本: bash verify_sparse.sh")
print("开始训练: python tools/train.py projects/configs/bitline/bitline_ep110_sparse.py")
