## Installation

### Requirements

- Linux or macOS with Python 3.8 recommended
- PyTorch and torchvision installed as a matching pair
- Detectron2 installed in the same environment
- OpenCV for demo and visualization commands
- CUDA-capable GPU for practical training and evaluation

### Example Conda Setup

```bash
conda create --name fastinst python=3.8 -y
conda activate fastinst
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python
```

Install Detectron2:

```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ..
```

Install optional dataset utilities:

```bash
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git
```

Install this repository:

```bash
git clone https://github.com/srirammadduri/fastinst-segmentation.git
cd fastinst-segmentation
pip install -r requirements.txt
```
