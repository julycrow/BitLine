#!/bin/bash
# CGNet完整DAM分析与可视化脚本
# 
# 使用方法:
#   bash run_dam_complete.sh [样本数] [输出前缀]
#
# 示例:
#   bash run_dam_complete.sh 500 dam_500    # 分析500样本
#   bash run_dam_complete.sh 100 dam_100    # 分析100样本
#   bash run_dam_complete.sh                # 默认500样本

set -e  # 遇到错误立即退出

# 获取脚本所在目录的项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
OUTPUT_DIR="$SCRIPT_DIR/outputs"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 默认参数
SAMPLES=${1:-500}
OUTPUT_PREFIX=${2:-"dam_${SAMPLES}_samples"}
JSON_FILE="${OUTPUT_DIR}/${OUTPUT_PREFIX}.json"
PLOT_DIR="${OUTPUT_DIR}/${OUTPUT_PREFIX}_plots"

echo "================================================================================"
echo "CGNet完整DAM分析与可视化"
echo "================================================================================"
echo ""
echo "配置:"
echo "  项目根目录: ${PROJECT_ROOT}"
echo "  输出目录: ${OUTPUT_DIR}"
echo "  样本数: ${SAMPLES}"
echo "  JSON输出: ${JSON_FILE}"
echo "  图片目录: ${PLOT_DIR}"
echo ""
echo "================================================================================"
echo ""

# 步骤1: 运行DAM分析
echo "步骤 1/3: 运行DAM分析..."
echo "--------------------------------------------------------------------------------"
python tools/dam_analysis/test_dam_analysis.py \
    --max-samples ${SAMPLES} \
    --output ${JSON_FILE} \
    --config projects/configs/cgnet/cgnet_ep110_dam.py \
    --checkpoint ckpts/cgnet_ep110.pth

if [ $? -ne 0 ]; then
    echo "❌ DAM分析失败！"
    exit 1
fi
echo ""
echo "✅ DAM分析完成！"
echo ""

# 步骤2: 生成可视化
echo "步骤 2/3: 生成可视化图表..."
echo "--------------------------------------------------------------------------------"
python tools/dam_analysis/plot_dam_distribution.py \
    --input ${JSON_FILE} \
    --output ${PLOT_DIR} \
    --bins 50

if [ $? -ne 0 ]; then
    echo "❌ 可视化生成失败！"
    exit 1
fi
echo ""
echo "✅ 可视化完成！"
echo ""

# 步骤3: 生成结果摘要
echo "步骤 3/3: 生成结果摘要..."
echo "--------------------------------------------------------------------------------"

# 提取关键统计信息
MEAN_RATIO=$(python -c "import json; d=json.load(open('${JSON_FILE}')); print(f\"{d['overall_statistics']['mean_reference_ratio']:.2%}\")")
LAYER0_RATIO=$(python -c "import json; d=json.load(open('${JSON_FILE}')); print(f\"{d['layer_wise_statistics']['layer_0']['mean_reference_ratio']:.2%}\")")
LAYER1_RATIO=$(python -c "import json; d=json.load(open('${JSON_FILE}')); print(f\"{d['layer_wise_statistics']['layer_1']['mean_reference_ratio']:.2%}\")")
TOTAL_SAMPLES=$(python -c "import json; d=json.load(open('${JSON_FILE}')); print(d['total_samples_analyzed'])")

echo ""
echo "================================================================================"
echo "分析结果摘要"
echo "================================================================================"
echo ""
echo "总样本数: ${TOTAL_SAMPLES}"
echo ""
echo "DAM统计:"
echo "  整体平均引用比例: ${MEAN_RATIO}"
echo "  Layer 0 (最稀疏):  ${LAYER0_RATIO}"
echo "  Layer 1 (最密集):  ${LAYER1_RATIO}"
echo ""
echo "生成的文件:"
echo "  JSON结果: ${JSON_FILE}"
echo "  图片目录: ${PLOT_DIR}/"
echo ""

# 列出生成的图片
echo "生成的图表:"
ls -lh ${PLOT_DIR}/*.png | awk '{printf "  - %s (%s)\n", $9, $5}'
echo ""

echo "================================================================================"
echo "✅ 完整分析流程成功完成！"
echo "================================================================================"
echo ""
echo "下一步:"
echo "  1. 查看JSON结果: cat ${JSON_FILE} | jq '.overall_statistics'"
echo "  2. 查看图片: ls -lh ${PLOT_DIR}/"
echo "  3. 打开图片查看器浏览可视化结果"
echo ""
