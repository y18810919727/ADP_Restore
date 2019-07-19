#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch


def fuse_imgs(imgs, attention, tools, device):

    batch_size, channels, height,  width = imgs.shape
    out_imgs = torch.zeros(imgs.shape, dtype=torch.float).to(device)
    for index, tool in enumerate(tools):
        img_res_intermediate = tool(imgs)
        cur_attention = attention[:, index].reshape(batch_size, 1, 1, 1).expand(
            img_res_intermediate.shape
        )
        img_res_intermediate = img_res_intermediate * cur_attention
        out_imgs = out_imgs + img_res_intermediate
    return out_imgs + imgs

