#!/bin/bash

# CGNet 参数量计算脚本

CONFIG_FILE="projects/configs/cgnet/cgnet_ep110.py"
DEPTH=3

echo "========================================"
echo "CGNet 模型参数量分析"
echo "========================================"
echo "配置文件: $CONFIG_FILE"
echo "统计深度: $DEPTH"
echo "========================================"
echo ""

# 运行参数量计算
python tools/analyze_flops_params/calculate_params.py $CONFIG_FILE --depth $DEPTH

echo ""
echo "========================================"
echo "如需查看更详细的模块信息，请运行:"
echo "python tools/analyze_flops_params/calculate_params.py $CONFIG_FILE --depth $DEPTH --detailed"
echo "========================================"
