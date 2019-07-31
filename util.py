#!/usr/bin/python
# -*- coding:utf8 -*-
import time
import numpy as np
import cv2
import math
import os
import json

import torch
import torch.utils.data as data
import h5py
import matplotlib.pyplot as plt
from config import CommonConfig


def h5extract(h5path=None, label_list=None):
    label_list = ['data', 'label'] if label_list is None else label_list
    f = h5py.File(h5path, 'r')
    data = dict()
    for key in label_list:
        data[key] = f[key][()].transpose(0,3,1,2) # dataset[()] is equal to dateset.value()
    f.close()
    return data

class MyDataset(data.Dataset):

    def __init__(self, data, label):
        super(MyDataset, self).__init__()
        self.data = data
        self.target = label

    def __getitem__(self, index):
        return (torch.from_numpy(self.data[index,:,:,:]).float(),
                torch.from_numpy(self.target[index,:,:,:]).float())

    def __len__(self):
        return self.data.shape[0]



def psnr_cal_single(im_input, im_label):
    loss = (im_input - im_label) ** 2
    eps = 1e-10
    loss_value = loss.mean() + eps
    psnr = 10 * math.log10(1.0 / loss_value)
    return psnr

def psnr_cal2(im_input, im_label, mean=True):
    return np.array([psnr_cal_single(im_input[i], im_label[i]) for i in range(im_input.shape[0])]).mean()

def psnr_cal(im_input, im_label, mean=True):
    if len(im_input.shape) == 3:
        im_input = np.expand_dims(im_input, axis=0)
        im_label = np.expand_dims(im_label, axis=0)
    loss = (im_input - im_label) ** 2
    eps = 1e-10
    loss_value = loss.mean(axis=(1,2,3)) + eps
    psnr = 10 * np.log10(1.0 / loss_value)
    if mean:
        return psnr.mean()
    else:
        return psnr

def cur_time_str():
    return time.strftime("%b_%d_%H:%M:%S_%Y", time.localtime())

def visual_img(img_array, save_dir=CommonConfig.tmp_dir):
    if len(img_array.shape) == 4:
        img_array=img_array.squeeze(0)
    if type(img_array) is torch.Tensor:
        img_array = img_array.detach().cpu().numpy()
    import imageio
    if img_array.shape[0]<=3:
        img_array = img_array.transpose(1,2,0)
    imageio.imsave(
        os.path.join(save_dir, 'tmp.png'),
        img_array
    )

def load_tools(tools_path, device):

    tools = []
    if 'tf2torch' in tools_path:
        import imp
        for tool_id in range(12):
            MainModel = imp.load_source('MainModel', os.path.join(tools_path, 'tool%02i.py' % (tool_id+1)))
            model_path = os.path.join(tools_path, 'tool%02i.pth' % (tool_id+1))
            model = torch.load(model_path)
            model.to(device)
            model.eval()
            tools.append(model)

    else:
        for tool_id in range(12):
            tool = torch.load(os.path.join(tools_path, 'tool%02i.pkd' % tool_id))
            if type(tool) is torch.nn.DataParallel:
                tool = tool.module
            tools.append(tool)
    return tools

def pngs_dir_read(pngs_path):

    path_in = [os.path.join(pngs_path, name) for name in os.listdir(pngs_path)]
    path_in.sort()
    imgs = load_imgs(path_in)
    return imgs

def load_imgs(list_in, size = 63):
    img_num = len(list_in)
    imgs = np.zeros([img_num, size, size, 3])
    for k in range(img_num):
        imgs[k, ...] = cv2.imread(list_in[k]) / 255.
    imgs = imgs.transpose([0, 3, 1, 2])
    return imgs
