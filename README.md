> Learnable Motion Coherence for Correspondence Pruning <br>
> Yuan Liu, Lingjie Liu, Cheng Lin, Zhen Dong, Wenping Wang <br>
> [Project Page](https://liuyuan-pal.github.io/LMCNet/)

Any questions or discussions are welcomed!

## Requirements & Compilation

1. Requirements

Required packages are listed in [requirements.txt](requirements.txt). 

The code is tested using Python-3.8.5 with PyTorch 1.7.1.

2. Compile extra modules

```shell script
cd network/knn_search
python setup.py build_ext --inplace
cd ../pointnet2_ext
python setup.py build_ext --inplace
cd ../../utils/extend_utils
python build_extend_utils_cffi.py
```
According to your installation path of CUDA, you may need to revise the variables cuda_version in [build_extend_utils_cffi.py](utils/extend_utils/build_extend_utils_cffi.py).

## Datasets & Pretrain Models

1. Download the YFCC100M dataset and the SUN3D dataset from the [OANet](https://github.com/zjhthu/OANet) repository and the ScanNet dataset from [here](https://drive.google.com/drive/folders/1E3Gibmtiqv9NZm5TxjxwgRPFTAM40FRV?usp=sharing).

2. Download pretrained LMCNet models from [here](https://drive.google.com/file/d/1mBXDxEKVsp3wGS5Xezpsc_oPlg-zgMDU/view?usp=sharing) and SuperGlue/SuperPoint models from [here](https://github.com/magicleap/SuperGluePretrainedNetwork/tree/master/models/weights).

3. Unzip and arrange all files like the following.
```
data/
├── superpoint/
    └── superpoint_v1.pth
├── superglue/
    ├── superglue_indoor.pth
    └── superglue_outdoor.pth
├── model/
    ├── lmcnet_sift_indoor/
    ├── lmcnet_sift_outdoor/
    └── lmcnet_spg_indoor/
├── yfcc100m/
├── sun3d_test/
├── sun3d_train/
├── scannet_dataset/
└── scannet_train_dataset/
```

## Evaluation

Evaluate on the YFCC100M with SIFT descriptors and Nearest Neighborhood (NN) matcher:
```shell script
python eval.py --name scannet --cfg configs/eval/lmcnet_sift_yfcc.yaml
```

Evaluate on the SUN3D with SIFT descriptors and NN matcher:
```shell script
python eval.py --name sun3d --cfg configs/eval/lmcnet_sift_sun3d.yaml
```

Evaluate on the ScanNet with SuperPoint descriptors and SuperGlue matcher:
```shell script
python eval.py --name scannet --cfg configs/eval/lmcnet_spg_scannet.yaml
```

## Training

1. Generate training dataset for training on YFCC100M with SIFT descriptor and NN matcher.
```shell script
python trainset_generate.py \
      --ext_cfg configs/detector/sift.yaml \
      --match_cfg configs/matcher/nn.yaml \
      --output data/yfcc_train_cache \
      --eig_name small_min \
      --prefix yfcc
```

2. Model training.
```shell script
python train_model.py --cfg configs/lmcnet/lmcnet_sift_outdoor_train.yaml
```

## Acknowledgement

We have used codes from the following repositories, and we thank the authors for sharing their codes.

SuperGlue: [https://github.com/magicleap/SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork)

OANet: [https://github.com/zjhthu/OANet](https://github.com/zjhthu/OANet)

KNN-CUDA: [https://github.com/vincentfpgarcia/kNN-CUDA](https://github.com/vincentfpgarcia/kNN-CUDA)

Pointnet2.PyTorch: [https://github.com/sshaoshuai/Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch)
