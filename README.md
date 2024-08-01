# 3D Point Cloud Central Axis Aggregation Network

## Environment

- Ubuntu 22.04
- Python 3.8
- Pytorch 2.1.0


```bash
conda create -n pointcept2 python=3.8 -y
conda activate pointcept2
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
# We use CUDA 11.8 and PyTorch 2.1.0 for our development of PTv3
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

cd libs/pointgroup_ops
python setup.py install
cd ../..


# PTv1 & PTv2 or precise eval
cd libs/pointops
# usual
python setup.py install

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu118  # choose version match your local cuda version

# Open3D (visualization, optional)
pip install open3d
```

FlashAttention

```bash
pip install flash-attn --no-build-isolation
```

## Dataset Prepare

Simulation Method of Point Cloud Data for Maize Populations：

1. Download: Physically Based Deformation of Single Maize Point Cloud Datasets \
    [link](https://www.kaggle.com/datasets/yangxin6/simulatio-maize-point-cloud-datasets)
2. run
```
python project/multi_gen_group_data_no_land.py
```


## Ground Truth Dataset
我们在上面4种传感器获得的共17个数据上进行了实验，数据目录如下
```
lidar__a.txt
lidar__c.txt
lidar__b.txt
lidar__d.txt
other__Maize-04_gt.txt
other__grou_maize_gd.txt
slam__slam_all.txt
rgb__0707_Tian_30_gt.txt
rgb__0707_502_30_gt.txt
rgb__0709_XY_20_gt.txt
rgb__0709_XY_30_gt.txt
rgb__0721_Tian_20_gt.txt
rgb__0729_Tian_30_gt.txt
uav__WasecaV5_clean.txt
uav__StPaulV6_clean.txt
uav__StPaulV3_clean.txt
uav__DjiV4_clean_gt.txt
```
测试数据的真值以及我们模型的预测结果我们公布在下面地址中：
datasets [link](https://www.kaggle.com/datasets/yangxin6/test-point-cloud-datasets-of-mazie-population)


此外，lidar、slam和rgb类型的数据是我们采集的数据，其他数据的来源如下：
- [other__grou_maize_gd]()
- [other__Maize-04_gt]()
- [uav__*]()

## Network

![3D-Point-Cloud-Central-Axis-Aggregation-Network.jpg](imgs%2F3D-Point-Cloud-Central-Axis-Aggregation-Network.jpg)

## Train

```bash
python tools/train.py --config-file configs/corn3d_group/insseg-pointgroup-v2m1-0-pt3-base_no_land.py
```

## Test
1. Change the `configs/corn3d_group/insseg-pointgroup-v2m1-0-pt3-base.py` `test=True` in `model` dict.

2. run
```bash
python tools/test.py --config-file configs/corn3d_group/insseg-pointgroup-v2m1-0-pt3-base.py  --options save_path="{weight_path}"  weight="{weight_path}/model_best.pth"
```
We provide our best model weights here: [model_pth](https://www.kaggle.com/datasets/yangxin6/3d-point-cloud-central-axis-aggregation-network)



## Reference
- [Pointcept](https://github.com/Pointcept/Pointcept)

## Citation

If you find this project useful in your research, please consider cite:

