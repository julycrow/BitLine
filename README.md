# BitLane
The implementation of BitLane.

## Abstract
End-to-end local high-definition (HD) map construction for autonomous driving requires both accurate centerline geometry and reliable graph topology reasoning. Existing centerline graph learning methods still face two main challenges. Topology prediction based on graph message passing can suffer from over-smoothing, and positive-subgraph supervision introduces sample-dependent graph dimensions that are mismatched with global generative modeling. To address these issues, we propose BitLine, a discrete bit diffusion framework for centerline graph learning in autonomous driving. BitLine reformulates topology prediction as discrete diffusion over a full adjacency matrix and introduces a full-graph denoising architecture with zero-padded full-graph supervision, enabling topology generation in a dimensionally consistent state space. In addition, a geometric smoothness regularization is introduced to stabilize centerline representations. Experiments on nuScenes and Argoverse2 show that BitLine achieves state-of-the-art performance. Extensive ablations further verify that discrete bit diffusion is better suited than Gaussian diffusion for binary topology modeling, and that stronger topology reasoning not only improves graph prediction quality but also benefits bottom-level detection.

## Usage

#### Installation
```bash
conda create -n bitlane-env python=3.8 -y
pip install -r requirement.txt

cd mmdetection3d
python setup.py develop

# Install GeometricKernelAttention (refer to MapTR)
cd projects/mmdet3d_plugin/bitline/modules/ops/geometric_kernel_attn
python setup.py build install
```

#### Data Preparation (nuScenes example)
```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes/ann --extra-tag nuscenes --version v1.0 --canbus ./data/nuscenes
```

## Acknowledgements
BitLane is built upon CGNet and related open-source projects.

Special thanks to **CGNet**:
[https://github.com/XiaoMi/CGNet](https://github.com/XiaoMi/CGNet)
