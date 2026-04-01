#!/bin/bash

# CGNet FLOPs计算脚本

CONFIG_FILE="projects/configs/cgnet/cgnet_ep110.py"
DEVICE="cuda"

echo "========================================"
echo "CGNet 模型FLOPs分析"
echo "========================================"
echo "配置文件: $CONFIG_FILE"
echo "计算设备: $DEVICE"
echo "========================================"
echo ""

# 检查依赖
echo "正在检查依赖..."
python -c "import fvcore" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ fvcore 已安装"
    METHOD="fvcore"
else
    python -c "import thop" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✓ thop 已安装"
        METHOD="thop"
    else
        echo "✗ 未检测到 fvcore 或 thop"
        echo "正在安装 fvcore..."
        pip install fvcore -q
        METHOD="fvcore"
    fi
fi

echo ""

# 运行FLOPs计算
python tools/analyze_flops_params/calculate_flops.py $CONFIG_FILE --device $DEVICE --method $METHOD

echo ""
echo "========================================"
echo "如需查看更详细的模块信息，请运行:"
echo "python tools/analyze_flops_params/calculate_flops.py $CONFIG_FILE --detailed --top-n 50"
echo "========================================"
