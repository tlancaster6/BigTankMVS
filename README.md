# BigTankMVS

## Installation
create and activate a conda env with python 3.12 (for open3d compatibility)
```commandline
conda create -n BigTankMVS python=3.12
conda activate BigTankMVS
```
install pytorch (modify for your system based on https://pytorch.org/get-started/locally/) and check gpu availability (should print True)
```commandline
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"
```
install remaining dependencies
```commandline
pip3 install kornia opencv-python open3d numpy toml tqdm
```