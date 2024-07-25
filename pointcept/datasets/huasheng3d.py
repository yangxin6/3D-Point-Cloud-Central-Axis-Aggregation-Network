#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/9 18:22
# @Author  : yangxin
# @Email   : yangxinnc@163.com
# @File    : corn3d_ins.py
# @Project : PlantPointSeg
# @Software: PyCharm
import glob
import os
from copy import deepcopy

import numpy as np
import torch
from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class Huasheng3dDataset(DefaultDataset):
    class2id = np.array([0, 1, 2])

    def __init__(
            self,
            split="train",
            data_root="data/huasheng3d",
            transform=None,
            test_mode=False,
            test_cfg=None,
            loop=1,
            ignore_index=-1,
    ):
        self.ignore_index = ignore_index
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        data = torch.load(data_path)
        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        scene_id = data["scene_id"]
        if "semantic_gt" in data.keys():
            segment = data["semantic_gt"].reshape([-1]).astype(int)
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1]).astype(int)
        else:
            instance = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
        )
        if "superpoint" in data.keys():
            superpoint = data["superpoint"]
            data_dict["superpoint"] = superpoint

        return data_dict

