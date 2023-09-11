# Self-Correlation and Cross-Correlation Learning for Few-Shot Remote Sensing Image Semantic Segmentation


Introduction
------------
This is the source code for our paper Self-Correlation and Cross-Correlation Learning for Few-Shot Remote Sensing Image Semantic Segmentation.

Network Architecture
------------
![network](architecture.png)

Training
------------
```
bash train.sh
```

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
