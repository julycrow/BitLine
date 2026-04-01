# CGNet DAM Analysis Tools

This directory contains all tools and documentation for DAM (Decoder cross-Attention Map) analysis on CGNet.

## Directory Structure

```
tools/dam_analysis/
├── README.md                           # This file
├── test_dam_analysis.py                # Main DAM analysis script
├── test_dam_quick.py                   # Quick test (10 samples)
├── plot_dam_distribution.py            # Visualization script
├── run_dam_analysis.sh                 # Full analysis script (500 samples)
├── run_dam_complete.sh                 # Complete workflow (analysis + visualization)
├── verify_dam_installation.py          # Installation verification
├── outputs/                            # Output directory (auto-created)
│   ├── *.json                         # Analysis results
│   ├── *_plots/                       # Visualization plots
│   └── ...
├── DAM_README.md                       # Documentation index
├── DAM_ANALYSIS_GUIDE.md               # Complete usage guide
├── DAM_VISUALIZATION_GUIDE.md          # Visualization guide
├── DAM_USAGE_QUICK_START_V2.md         # Quick start guide
├── DAM_UPDATE_LOG.md                   # Update changelog
└── PLOT_LABELS_UPDATE.md               # Plot labels update notes
```

## Quick Start

### 1. Quick Test (10 samples, ~10 seconds)

```bash
# From project root
cd /path/to/CGNet
python tools/dam_analysis/test_dam_quick.py
```

Output: `tools/dam_analysis/outputs/dam_quick_test.json`

### 2. Standard Analysis (500 samples, ~5-8 minutes)

```bash
# From project root
cd /path/to/CGNet
bash tools/dam_analysis/run_dam_analysis.sh
```

Output: `tools/dam_analysis/outputs/dam_analysis_results.json`

### 3. Complete Workflow (Analysis + Visualization)

```bash
# From project root
cd /path/to/CGNet
bash tools/dam_analysis/run_dam_complete.sh 500 my_analysis
```

Output:
- `tools/dam_analysis/outputs/my_analysis.json`
- `tools/dam_analysis/outputs/my_analysis_plots/` (8 PNG images)

### 4. Custom Analysis

```bash
# From project root
cd /path/to/CGNet
python tools/dam_analysis/test_dam_analysis.py \
    --config projects/configs/cgnet/cgnet_ep110_dam.py \
    --checkpoint ckpts/cgnet_ep110.pth \
    --max-samples 1000 \
    --output tools/dam_analysis/outputs/custom_analysis.json
```

### 5. Generate Visualizations

```bash
# From project root
cd /path/to/CGNet
python tools/dam_analysis/plot_dam_distribution.py \
    --input tools/dam_analysis/outputs/dam_analysis_results.json \
    --output tools/dam_analysis/outputs/my_plots
```

## Key Features

### Analysis (`test_dam_analysis.py`)
- Analyzes decoder cross-attention maps
- Calculates DAM ratio (% of encoder tokens referenced)
- Per-layer statistics
- Sample-level data for visualization
- JSON output with complete statistics

### Visualization (`plot_dam_distribution.py`)
- Histogram plots per decoder layer
- Combined comparison plot (2x3 layout)
- Boxplot for statistical comparison
- English labels (no Chinese font required)
- High-resolution PNG output (300 DPI)

### Complete Workflow (`run_dam_complete.sh`)
- One command for full pipeline
- Automatic analysis + visualization
- Summary statistics output
- Customizable sample count

## Results Interpretation

### Typical Results (CGNet on nuScenes)
- **Overall DAM**: ~57% (vs. Sparse DETR ~45%)
- **Layer 0**: ~37% (most sparse - key region focus)
- **Layer 1-5**: ~58-62% (denser - global reasoning)

### What This Means
- **Layer 0 sparsity**: Indicates efficient selective attention
- **64% unused tokens**: Optimization potential
- **Layer-wise variation**: Normal hierarchical attention pattern

## Output Files

### JSON Results
```json
{
  "layer_wise_statistics": {
    "layer_0": {
      "mean_reference_ratio": 0.37,
      "std_reference_ratio": 0.01,
      ...
    }
  },
  "overall_statistics": {
    "mean_reference_ratio": 0.57,
    ...
  },
  "sample_level_data": {
    "layer_0": [0.356, 0.372, ...],
    "layer_1": [0.599, 0.618, ...],
    ...
  },
  "total_samples_analyzed": 500
}
```

### Visualization Plots
1. `layer_0_dam_distribution.png` - Layer 0 histogram
2. `layer_1_dam_distribution.png` - Layer 1 histogram
3. ... (one per layer, 6 total)
4. `all_layers_dam_distribution.png` - Combined comparison
5. `layers_comparison_boxplot.png` - Statistical boxplot

## Documentation

- **[DAM_README.md](DAM_README.md)** - Documentation index
- **[DAM_ANALYSIS_GUIDE.md](DAM_ANALYSIS_GUIDE.md)** - Complete usage guide
- **[DAM_VISUALIZATION_GUIDE.md](DAM_VISUALIZATION_GUIDE.md)** - Visualization details
- **[DAM_USAGE_QUICK_START_V2.md](DAM_USAGE_QUICK_START_V2.md)** - Quick start
- **[DAM_UPDATE_LOG.md](DAM_UPDATE_LOG.md)** - Update changelog

## Requirements

- Python 3.8+
- PyTorch 1.9+
- mmdetection3d
- matplotlib
- numpy
- nuScenes dataset
- Trained CGNet checkpoint

## Verification

Check if everything is installed correctly:

```bash
# From project root
cd /path/to/CGNet
python tools/dam_analysis/verify_dam_installation.py
```

Expected output:
```
✓ All core files found
✓ All modules importable
✓ DAM analysis system ready!
```

## Tips

1. **Start with quick test**: Use `test_dam_quick.py` first
2. **Check GPU memory**: 500 samples needs ~8GB VRAM
3. **Outputs location**: All outputs go to `outputs/` subdirectory
4. **Visualization only**: Can run plots without re-analysis
5. **Custom bins**: Use `--bins 100` for finer histograms

## Troubleshooting

### "Module not found" error
```bash
# Make sure you're running from project root
cd /path/to/CGNet
python tools/dam_analysis/test_dam_analysis.py
```

### "File not found" error
```bash
# Check if paths are correct
ls tools/dam_analysis/outputs/
```

### "CUDA out of memory"
```bash
# Reduce sample count
python tools/dam_analysis/test_dam_analysis.py --max-samples 100
```

## Examples

### Example 1: Quick validation
```bash
cd /path/to/CGNet
python tools/dam_analysis/test_dam_quick.py
# Output: tools/dam_analysis/outputs/dam_quick_test.json
```

### Example 2: Full analysis with visualization
```bash
cd /path/to/CGNet
bash tools/dam_analysis/run_dam_complete.sh 1000 full_analysis
# Output: 
#   tools/dam_analysis/outputs/full_analysis.json
#   tools/dam_analysis/outputs/full_analysis_plots/*.png
```

### Example 3: Visualization only
```bash
cd /path/to/CGNet
python tools/dam_analysis/plot_dam_distribution.py \
    --input tools/dam_analysis/outputs/dam_quick_test.json \
    --output tools/dam_analysis/outputs/quick_plots \
    --bins 100
```

## Citation

If you use this DAM analysis tool, please cite:

```bibtex
@article{roh2021sparse,
  title={Sparse DETR: Efficient End-to-End Object Detection with Learnable Sparsity},
  author={Roh, Byungseok and Shin, JaeWoong and Shin, Wuhyun and Kim, Saehoon},
  journal={arXiv preprint arXiv:2111.14330},
  year={2021}
}
```

## Contact

For issues or questions:
1. Check documentation in this directory
2. Verify installation with `verify_dam_installation.py`
3. Review troubleshooting section above

---

**Happy analyzing!** 🚀📊
