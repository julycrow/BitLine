#!/bin/bash

# CGNet 模型综合分析脚本
# 同时计算参数量和FLOPs

CONFIG_FILE="projects/configs/cgnet/cgnet_ep110.py"
OUTPUT_DIR="model_analysis"

echo "========================================"
echo "CGNet 模型综合复杂度分析"
echo "========================================"
echo "配置文件: $CONFIG_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "========================================"
echo ""

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "1. 正在分析参数量..."
python tools/analyze_flops_params/calculate_params.py $CONFIG_FILE --detailed > ${OUTPUT_DIR}/params_analysis.txt
python tools/analyze_flops_params/calculate_params.py $CONFIG_FILE | tee ${OUTPUT_DIR}/params_summary.txt

echo ""
echo "2. 正在分析FLOPs..."
python tools/analyze_flops_params/calculate_flops.py $CONFIG_FILE --detailed > ${OUTPUT_DIR}/flops_analysis.txt  
python tools/analyze_flops_params/calculate_flops.py $CONFIG_FILE | tee ${OUTPUT_DIR}/flops_summary.txt

echo ""
echo "========================================"
echo "分析完成!"
echo "========================================"
echo "结果已保存到 $OUTPUT_DIR/ 目录:"
echo "  - params_summary.txt  : 参数量摘要"
echo "  - params_analysis.txt : 参数量详细分析"
echo "  - flops_summary.txt   : FLOPs摘要"
echo "  - flops_analysis.txt  : FLOPs详细分析"
echo "========================================"
