# This document shows how to calculate eigen vectors

1. Extract features
------------
```
python extract.py extract_features --images_list "${DataRoot}/images.txt" --images_root "${DataRoot}/images" --output_dir "${DataRoot}/features/resnet50" --model_name resnet50 --batch_size 1
```

2. Calculate eigen vectors
------------
```
python extract.py extract_eigs --images_root "${DataRoot}/images" --features_dir "${DataRoot}/features/resnet101" --which_matrix "laplacian" --output_dir "${DataRoot}/eigs/resnet101/laplacian_top5_c5_l26" --K 5 --image_downsample_factor 4 --image_color_lambda 5
```
