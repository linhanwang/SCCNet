# Self-Correlation and Cross-Correlation Learning for Few-Shot Remote Sensing Image Semantic Segmentation


Introduction
------------
This is the source code for our paper Self-Correlation and Cross-Correlation Learning for Few-Shot Remote Sensing Image Semantic Segmentation.

Network Architecture
------------
![network](architecture.png)

### Installation
* Install PyTorch 2.0.1 and other dependenies
* Clone this repo

```
git clone https://github.com/linhanwang/SCCNet.git
```

### Data Preparation

Download data from [here](), unzip and put it under your directory 'SCCNet'.


### Train

```
python train.py  --max_steps 200000 --freeze True --datapath './remote_sensing/iSAID_patches' --img_size 256 --backbone resnet50 --fold 0 --benchmark isaid --lr 9e-4 --bsz 32 --logpath exp_name
```

The log and checkpoints are stored under directory 'logs'.

## Testing
```
bash test.sh
```

## Spectral Segmentation
```
cd spectral

# extract feature from backbone
bash extract_features.sh

# calculate eigenvectors
bash extract_eigs.sh
```

The fusion process is implemented in test.py, you can turn it on in test.sh.

# Data
[iSAID-5^i](https://github.com/caoql98/SDM)
[DLRSD](https://sites.google.com/view/zhouwx/dataset#h.p_hQS2jYeaFpV0)


# Reference
We borrow code from public projects [SDM](https://github.com/caoql98/SDM), [HSNet](https://github.com/juhongm999/hsnet), [dss](https://github.com/lukemelas/deep-spectral-segmentation).
