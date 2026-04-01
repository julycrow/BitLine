# CGNet 模型复杂度分析工具

本目录包含用于分析CGNet模型参数量和FLOPs的完整工具集。

## 📁 目录结构

```
tools/analyze_flops_params/
├── README.md                           # 本文档
├── calculate_params.py                 # 参数量计算脚本
├── calculate_flops.py                  # FLOPs计算脚本
├── compare_params_flops.py             # 参数量与FLOPs对比工具
├── analyze_params.sh                   # 参数量分析快捷脚本
├── analyze_flops.sh                    # FLOPs分析快捷脚本
├── analyze_model_complexity.sh         # 综合分析快捷脚本
├── 参数量分析使用说明.md                # 参数量分析详细说明
├── FLOPs分析使用说明.md                # FLOPs分析详细说明
├── README_参数量分析.md                 # 参数量分析参考文档
├── README_模型复杂度分析.md             # 模型复杂度分析概述
└── 模型复杂度快速参考.md                # 快速参考指南
```

## 🚀 快速开始

### 方式一：使用Shell脚本（推荐）

```bash
# 1. 仅分析参数量
bash tools/analyze_flops_params/analyze_params.sh

# 2. 仅分析FLOPs
bash tools/analyze_flops_params/analyze_flops.sh

# 3. 综合分析（同时计算参数量和FLOPs，并保存到文件）
bash tools/analyze_flops_params/analyze_model_complexity.sh

# 4. 并排对比参数量和FLOPs
python tools/analyze_flops_params/compare_params_flops.py
```

### 方式二：直接使用Python脚本

```bash
# 1. 计算参数量
python tools/analyze_flops_params/calculate_params.py projects/configs/cgnet/cgnet_ep110.py

# 2. 计算FLOPs
python tools/analyze_flops_params/calculate_flops.py projects/configs/cgnet/cgnet_ep110.py

# 3. 详细模式（显示更多模块信息）
python tools/analyze_flops_params/calculate_params.py projects/configs/cgnet/cgnet_ep110.py --detailed
python tools/analyze_flops_params/calculate_flops.py projects/configs/cgnet/cgnet_ep110.py --detailed --top-n 50
```

## 📊 分析结果示例

### 参数量分析结果
```
========================================================================================================================
                                                      参数量分析摘要                                                       
========================================================================================================================
总参数量:                       39,028,513 (39.03M)
========================================================================================================================

========================================================================================================================
                                            主要组件参数量分布 (与FLOPs层级一致)                                            
========================================================================================================================
组件名称                                                                              参数量                        占比
------------------------------------------------------------------------------------------------------------------------
img_backbone                                                                      23.51M                    60.23%
transformer.decoder                                                                4.10M                    10.51%
pts_bbox_head                                                                      3.76M                     9.64%
img_neck                                                                           2.10M                     5.37%
transformer.encoder                                                              641.53K                     1.64%
bev_embedding                                                                    327.68K                     0.84%
positional_encoding                                                              327.68K                     0.84%
========================================================================================================================
```

### FLOPs分析结果
```
========================================================================================================================
                                                       FLOPs 分析摘要                                                       
========================================================================================================================
总FLOPs:                   151,723,878,400 (151.72G)
========================================================================================================================

========================================================================================================================
                                                      主要组件FLOPs分布                                                       
========================================================================================================================
组件名称                                                                                  FLOPs                        占比
------------------------------------------------------------------------------------------------------------------------
img_backbone                                                                         88.03G                    58.02%
transformer.decoder                                                                  33.87G                    22.32%
img_neck                                                                             20.05G                    13.22%
transformer.encoder                                                                   9.34G                     6.16%
reg_branches                                                                        196.61M                     0.13%
vertex_inteact                                                                      163.84M                     0.11%
========================================================================================================================
```

## 🔧 高级选项

### calculate_params.py 选项

```bash
python tools/analyze_flops_params/calculate_params.py --help

参数:
  config              配置文件路径
  --depth DEPTH       统计深度 (default: 3)
  --detailed          显示详细的子模块参数信息
```

### calculate_flops.py 选项

```bash
python tools/analyze_flops_params/calculate_flops.py --help

参数:
  config              配置文件路径
  --device {cuda,cpu} 运行设备 (default: cuda)
  --detailed          显示详细的模块FLOPs信息
  --top-n TOP_N       显示Top N的模块 (default: 30)
  --method {auto,custom,fvcore,thop}
                      FLOPs计算方法 (default: auto, 推荐custom)
```

## 📈 关键发现

基于CGNet ep110模型的分析：

### 参数量分布
- **img_backbone (ResNet50)**: 23.51M (60.23%) - 最大的参数贡献者
- **transformer.decoder**: 4.10M (10.51%)
- **img_neck (FPN)**: 2.10M (5.37%)
- **transformer.encoder**: 641.53K (1.64%)

### FLOPs分布
- **img_backbone**: 88.03G (58.02%) - 主要计算负载
- **transformer.decoder**: 33.87G (22.32%) - 第二大计算负载
- **img_neck**: 20.05G (13.22%)
- **transformer.encoder**: 9.34G (6.16%)

### 优化建议
1. **Backbone优化**: 占据60%参数和58%计算量，可考虑轻量化backbone
2. **Transformer优化**: Decoder的计算量占22%，可优化attention机制
3. **模型压缩**: 可考虑知识蒸馏或量化来减小模型体积

## 📚 详细文档

- `参数量分析使用说明.md` - 参数量计算的详细说明
- `FLOPs分析使用说明.md` - FLOPs计算的详细说明
- `README_模型复杂度分析.md` - 模型复杂度分析的完整指南
- `模型复杂度快速参考.md` - 常用命令快速参考

## ⚙️ 依赖要求

```bash
# 基础依赖（必需）
torch
mmcv
mmdet3d

# FLOPs计算依赖（可选，自动使用custom方法）
pip install fvcore  # 推荐
# 或
pip install thop
```

## 🔍 注意事项

1. **Transformer层级**: transformer已细分为encoder和decoder两部分统计
2. **计算方法**: FLOPs默认使用custom方法，针对CGNet优化，更准确
3. **设备选择**: FLOPs计算建议使用CUDA加速（如果可用）
4. **配置文件**: 所有脚本默认使用`projects/configs/cgnet/cgnet_ep110.py`

## 🛠️ 自定义分析

修改shell脚本中的配置：

```bash
# analyze_params.sh
CONFIG_FILE="projects/configs/cgnet/your_config.py"
DEPTH=3

# analyze_flops.sh
CONFIG_FILE="projects/configs/cgnet/your_config.py"
DEVICE="cuda"
```

## 📞 问题反馈

如遇到问题，请检查：
1. 配置文件路径是否正确
2. Python环境是否正确激活
3. 必要的依赖是否已安装
4. CUDA是否可用（对于GPU分析）

---

*最后更新: 2025年12月1日*
