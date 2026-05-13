# FastInst Segmentation

FastInst Segmentation is a portfolio-ready computer vision repository for real-time instance segmentation on COCO-style datasets. It packages a Detectron2-based FastInst workflow with training, evaluation, demo, dataset preparation, and model-analysis utilities.

The core model is based on the FastInst architecture from the paper [FastInst: A Simple Query-Based Model for Real-Time Instance Segmentation](https://arxiv.org/abs/2303.08594). This repository focuses on making that workflow easy to run, inspect, and extend for practical experimentation.

<p align="center"><img width="100%" src="figures/fastinst.png" /></p>

## Problem Statement

Instance segmentation systems need to identify every object in an image and produce a pixel-level mask for each object. High-quality methods can be expensive to run, while lightweight models often lose mask quality. This project explores FastInst as a practical balance: query-based segmentation with real-time inference characteristics and reproducible COCO-style training/evaluation commands.

## What I Built

- A clean FastInst/Detectron2 project layout for COCO instance segmentation experiments.
- Training and evaluation entry points through `train_net.py`.
- Configurations for ResNet-50, ResNet-101, and ResNet-50d-DCN FastInst variants.
- Dataset preparation docs and utilities for COCO, ADE20K, and panoptic/semantic segmentation assets.
- Demo tooling for image, video, and webcam inference.
- A COCO annotation sampling utility for quickly inspecting dataset records before running full experiments.

## Tech Stack

- Python
- PyTorch
- Detectron2
- CUDA-enabled training/inference
- OpenCV for visualization demos
- COCO, ADE20K, and Cityscapes-style dataset formats

## Architecture

```text
.
├── configs/                 # Model and dataset config files
├── datasets/                # Dataset setup docs and conversion utilities
├── demo/                    # Image/video/webcam inference demo
├── fastinst/                # FastInst model, data mappers, evaluators, and config
├── tools/                   # Analysis, conversion, evaluation, and dataset sampling scripts
├── train_net.py             # Main Detectron2 training/evaluation entry point
├── INSTALL.md               # Environment setup notes
└── requirements.txt         # Python package dependencies beyond PyTorch/Detectron2
```

The training flow follows Detectron2 conventions: configs define the model, dataset mapper, solver, and evaluation behavior; `train_net.py` builds the trainer/evaluator; `fastinst/` contains model components such as the pixel decoder, transformer decoder, matcher, criterion, and inference logic.

## Quick Local Run

Create an environment and install dependencies:

```bash
conda create --name fastinst python=3.8 -y
conda activate fastinst
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python
pip install -r requirements.txt
```

Install Detectron2 in the same environment:

```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ..
```

Clone this repository:

```bash
git clone https://github.com/srirammadduri/fastinst-segmentation.git
cd fastinst-segmentation
```

Inspect a small slice of COCO annotations:

```bash
python tools/sample_coco_data.py --download
```

Run a pretrained checkpoint on the demo script:

```bash
python demo/demo.py \
  --config-file configs/coco/instance-segmentation/fastinst_R50_ppm-fpn_x1_576.yaml \
  --input path/to/image.jpg \
  --output demo-output \
  --opts MODEL.WEIGHTS path/to/checkpoint.pth
```

## Full Experiment Workflow

Set the dataset root:

```bash
export DETECTRON2_DATASETS=/path/to/datasets
```

Prepare COCO in the expected Detectron2 layout:

```text
$DETECTRON2_DATASETS/
└── coco/
    ├── annotations/
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    ├── train2017/
    └── val2017/
```

Evaluate a pretrained FastInst checkpoint:

```bash
python train_net.py \
  --eval-only \
  --num-gpus 1 \
  --config-file configs/coco/instance-segmentation/fastinst_R50_ppm-fpn_x1_576.yaml \
  MODEL.WEIGHTS path/to/checkpoint.pth
```

Train from a config:

```bash
python train_net.py \
  --num-gpus 1 \
  --config-file configs/coco/instance-segmentation/fastinst_R50_ppm-fpn_x1_576.yaml
```

Scale to multi-GPU training by increasing `--num-gpus` and using the same config. Outputs are written to the `OUTPUT_DIR` specified in each config.

Analyze model cost:

```bash
python tools/analyze_model.py \
  --num-inputs 100 \
  --tasks flop \
  --config-file configs/coco/instance-segmentation/fastinst_R50_ppm-fpn_x1_576.yaml
```

## Results

No local training logs or custom benchmark artifacts are included in this repository. The table below records published FastInst COCO instance segmentation reference metrics so the expected performance envelope is clear and reproducible with the linked checkpoints.

| Model | Backbone | Epochs | Input | AP val | AP | Params | GFlops | FPS V100 | Checkpoint |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FastInst-D1 | R50 | 50 | 576 | 34.9 | 35.6 | 30M | 49.6 | 53.8 | [model](https://github.com/junjiehe96/FastInst/releases/download/v0.1.0/fastinst_R50_ppm-fpn_x1_576_34.9.pth) |
| FastInst-D3 | R50 | 50 | 640 | 37.9 | 38.6 | 34M | 75.5 | 35.5 | [model](https://github.com/junjiehe96/FastInst/releases/download/v0.1.0/fastinst_R50_ppm-fpn_x3_640_37.9.pth) |
| FastInst-D3 | R101 | 50 | 640 | 38.9 | 39.9 | 53M | 112.9 | 28.0 | [model](https://github.com/junjiehe96/FastInst/releases/download/v0.1.0/fastinst_R101_ppm-fpn_x3_640_38.9.pth) |
| FastInst-D1 | R50-d-DCN | 50 | 576 | 37.4 | 38.0 | 30M | - | 47.8 | [model](https://github.com/junjiehe96/FastInst/releases/download/v0.1.0/fastinst_R50-vd-dcn_ppm-fpn_x1_576_37.4.pth) |
| FastInst-D3 | R50-d-DCN | 50 | 640 | 40.1 | 40.5 | 35M | - | 32.5 | [model](https://github.com/junjiehe96/FastInst/releases/download/v0.1.0/fastinst_R50-vd-dcn_ppm-fpn_x3_640_40.1.pth) |

For a lightweight reproducibility path without running full training, use:

```bash
python tools/sample_coco_data.py --download
python train_net.py --eval-only --num-gpus 1 \
  --config-file configs/coco/instance-segmentation/fastinst_R50_ppm-fpn_x1_576.yaml \
  MODEL.WEIGHTS path/to/fastinst_R50_ppm-fpn_x1_576_34.9.pth
```

## Reference Material

This repository does not include a separate report or paper folder. The FastInst paper remains the primary architectural reference:

```bibtex
@article{he2023fastinst,
  title={FastInst: A Simple Query-Based Model for Real-Time Instance Segmentation},
  author={He, Junjie and Li, Pengyu and Geng, Yifeng and Xie, Xuansong},
  journal={arXiv preprint arXiv:2303.08594},
  year={2023}
}
```

## License and Attribution

FastInst is released under the [MIT License](LICENSE). This repository keeps the original FastInst license, paper citation, and relevant source attributions. It also builds on the Detectron2, DETR, and Mask2Former ecosystems.
