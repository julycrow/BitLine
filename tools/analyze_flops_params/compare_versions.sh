#!/bin/bash

# CGNet普通版本vs稀疏版本对比分析

echo "========================================"
echo "CGNet 普通版本 vs 稀疏版本 对比分析"
echo "========================================"
echo ""

# 运行对比脚本
python tools/analyze_flops_params/compare_normal_sparse.py

echo ""
echo "========================================"
echo "如需查看单独版本的详细信息:"
echo ""
echo "  普通版本参数量:"
echo "    python tools/analyze_flops_params/calculate_params.py projects/configs/cgnet/cgnet_ep110.py"
echo ""
echo "  普通版本FLOPs:"
echo "    python tools/analyze_flops_params/calculate_flops.py projects/configs/cgnet/cgnet_ep110.py"
echo ""
echo "  稀疏版本参数量:"
echo "    python tools/analyze_flops_params/calculate_params.py projects/configs/cgnet/cgnet_ep110_sparse.py"
echo ""
echo "  稀疏版本FLOPs:"
echo "    python tools/analyze_flops_params/calculate_flops.py projects/configs/cgnet/cgnet_ep110_sparse.py"
echo "========================================"
