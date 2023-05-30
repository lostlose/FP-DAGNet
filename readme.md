# FP_DAGNet
This code is a demo of our paper ICCV 9721. Currently we only provide code for testing, the training code will be released to public after the acceptance of our paper.

# Dataset
To proceed, please download the Gen1 dataset on your own. Download [here](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/).

# Environment
```
1. Python 3.8.*
2. CUDA 11.1
3. PyTorch 
4. TorchVision 
5. fitlog
```

# Install
Create a  virtual environment and activate it.
```shell
conda create -n FP_DAGNet python=3.8
conda activate FP_DAGNet
```
The code has been tested with PyTorch 1.8.1 and Cuda 11.1.
```shell
conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=11.1 -c pytorch
conda install matplotlib tqdm pycocotools
conda install tensorboard tensorboardX
conda install scipy scikit-image opencv 
```

# Cofe for FP_DAGNet
We provide data pre-process and test code for Gen1.

## Pre-process
For sbt procedure, execute: \
  `python data/sbt_frame.py --root xxx -t 3 -fs 3 -tf 20`


## Test
For test, execute: \
  `bash test.sh`

  
# Reference
https://github.com/Huawei-BIC
https://github.com/yjh0410/PyTorch_YOLO-Family