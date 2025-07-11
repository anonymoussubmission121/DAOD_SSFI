## Installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do
conda create --name daod-ssfi -y
source /opt/conda/etc/profile.d/conda.sh
conda activate daod-ssfi

# this installs the right pip and dependencies for the fresh python
conda install python=3.9.7 ipython=8.1.1 pip=21.2.4 -y

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 11.3
conda install pytorch=1.12.0 torchvision=0.13.0 cudatoolkit=11.3 -c pytorch -y

# maskrcnn_benchmark and coco api dependencies
pip install piqa ninja==1.10.2.3 yacs==0.1.8 cython==0.29.28 matplotlib==3.5.1 tqdm==4.63.0 opencv-python==4.5.5.64 numpy==1.22.3 torchfile pycocotools==2.0.2

# MIC dependencies
pip install timm==0.6.11 kornia==0.5.8 einops==0.4.1 torchviz

cd DAOD_SSFI/det
# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop


pip install h5py==3.6.0 scipy==1.8.0
```
