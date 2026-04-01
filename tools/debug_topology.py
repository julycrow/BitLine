#!/usr/bin/env python
"""诊断topology评估卡住的问题"""

import pickle
import sys
import os
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import numpy as np

# 添加路径
sys.path.append('/home/ubuntu/disk4/jmk3/Project/CGNet')

from projects.mmdet3d_plugin.datasets.topo_eval.topo_evaluator import GraphEvaluator

def test_single_sample(evaluator, annotation, prediction, idx):
    """测试单个样本,带超时和异常处理"""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Sample {idx} timeout!")
    
    # 设置30秒超时
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    
    try:
        result = evaluator.evaluate_graph(annotation, prediction)
        signal.alarm(0)  # 取消超时
        print(f"Sample {idx}: OK ({result.get('APLS', 'N/A'):.3f})")
        return result
    except TimeoutError as e:
        print(f"Sample {idx}: TIMEOUT (>30s)")
        return None
    except Exception as e:
        print(f"Sample {idx}: ERROR - {e}")
        return None

def main():
    # 加载数据
    ann_file = '/home/ubuntu/disk4/jmk3/Project/CGNet/data/nuscenes/ann/nuscenes_graph_anns_val.pkl'
    result_file = '/home/ubuntu/disk4/jmk3/Project/CGNet/work_dirs/cgnet_ep24_fp16/Wed_Dec__4_18_57_19_2024/nuscenes_graph_results.pkl'
    
    print(f"Loading annotations from: {ann_file}")
    with open(ann_file, 'rb') as f:
        annotations = pickle.load(f)
    
    print(f"Loading results from: {result_file}")
    with open(result_file, 'rb') as f:
        predictions = pickle.load(f)
    
    print(f"Total samples: {len(annotations)}")
    
    # 创建evaluator
    evaluator = GraphEvaluator(radius=5, interp_dist=2.5, prop_dist=80, gmode='direct')
    
    # 逐个测试前20个样本,找出慢的样本
    print("\n=== Testing individual samples ===")
    slow_samples = []
    for i in range(min(20, len(annotations))):
        import time
        start = time.time()
        result = test_single_sample(evaluator, annotations[i], predictions[i], i)
        elapsed = time.time() - start
        if elapsed > 5:
            slow_samples.append((i, elapsed))
        if result is None:
            print(f"  ^ This sample is problematic!")
    
    if slow_samples:
        print(f"\n=== Slow samples (>5s) ===")
        for idx, elapsed in slow_samples:
            print(f"Sample {idx}: {elapsed:.1f}s")
    
    print("\nDone!")

if __name__ == '__main__':
    main()
