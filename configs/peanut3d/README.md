# 3D Peanut Instance Segmentation


## Train

```bash
python tools/train.py --config-file configs/peanut3d/insseg-pointgroup-v3m1-0-pt3.py
```

## Test
1. Change the `configs/peanut3d/insseg-pointgroup-v3m1-0-pt3.py` `test=True` in `model` dict.

2. run
```bash
python tools/test.py --config-file configs/peanut3d/insseg-pointgroup-v3m1-0-pt3.py  --options save_path="{weight_path}"  weight="{weight_path}/model_best.pth"
```
