#!/bin/bash
# 快速测试topology评估,找出卡住的位置

cd /home/ubuntu/disk4/jmk3/Project/CGNet

echo "=== Testing topology evaluation with detailed logs ==="
echo "Starting at: $(date)"

export CUDA_VISIBLE_DEVICES=3

# 运行测试,只评估topology,带超时
timeout 600 python -u tools/test.py \
    projects/configs/cgnet/cgnet_ep24.py \
    /home/ubuntu/disk4/jmk3/Project/CGNet/work_dirs/cgnet_ep24_fp16/epoch_24.pth \
    --eval topology \
    2>&1 | grep -E "Batch|Processing|completed|WARNING|ERROR|Traceback" || echo "Test completed or timed out"

echo "Finished at: $(date)"
