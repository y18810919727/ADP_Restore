#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch

import os
from util import h5extract
class ImageGenerator(object):
    def __init__(self, train_dir, shuffle=True):
        self.train_list = [os.path.join(train_dir, file) for file in os.listdir(train_dir) if file.endswith('h5')]
        self.train_file_name = [file for file in os.listdir(train_dir) if file.endswith('h5')]
        self.train_max = len(self.train_list)
        self.data_index = 0
        self.data_len = 0
        self.shuffle = shuffle
        if self.train_max == 0:
            raise ValueError('No .h5 file found')

        self.file_index = -1
        self.update_file()
        self.img_shape = self.data[0].shape

    def update_file(self):
        self.file_index = (self.file_index+1) % self.train_max
        self.data_index = 0
        h5_data = h5extract(self.train_list[self.file_index])
        self.data, self.label = h5_data['data'], h5_data['label']

        self.data_len = len(self.data)
        if self.shuffle is True:
            id_list = np.arange(self.data_len)
            np.random.shuffle(id_list)
            self.data = self.data[id_list]
            self.label = self.label[id_list]

    def generate_images(self, num):
        cur_cnt = 0
        res_data = np.zeros((num, ) + self.img_shape) # tuple + tuple = tuple
        res_label = np.zeros((num, ) + self.img_shape)

        while cur_cnt < num:
            tmp_cnt = min(self.data_len - self.data_index, num - cur_cnt)
            res_data[cur_cnt:cur_cnt + tmp_cnt] = self.data[self.data_index: self.data_index + tmp_cnt]
            res_label[cur_cnt:cur_cnt + tmp_cnt] = self.label[self.data_index: self.data_index + tmp_cnt]
            cur_cnt += tmp_cnt
            self.data_index += tmp_cnt
            if self.data_index >= self.data_len:
                self.update_file()
            if cur_cnt >= num:
                break
        return res_data, res_label

    def generate_all(self):
        data_list = []
        begin_file_index = self.file_index
        begin_data_index = self.data_index
        while True:
            data_list.append(self.generate_images(self.data_len-self.data_index))
            if self.file_index == begin_file_index:
                data_list.append(self.generate_images(begin_data_index))
                break
        res = tuple([np.concatenate(item) for item in zip(*data_list)])
        return res

    def restore_state(self, file_index, data_index):
        if file_index >= len(self.train_list):
            raise ValueError('Can not reset state, file_index is bigger than the numbers of h5 files')
        self.file_index = file_index - 1
        self.update_file()
        if data_index >= self.data_len:
            raise ValueError('Can not reset state, data_index is bigger than the maximum index of data')
        self.data_index = data_index










    def __str__(self):
        return 'file_index:{}, data_index:{}, file_name:{}'.format(self.file_index, self.data_index,
                                                                   self.train_file_name[self.file_index])


if __name__ == '__main__':
    #img_generator = ImageGenerator('../data/tool_data/validation')
    #img_generator = ImageGenerator('../data/validation/')
    img_generator = ImageGenerator('../data/train/')

    print(img_generator.generate_all()[0].shape)
    print(img_generator)

    print(img_generator.generate_images(10)[0].shape)
    print(img_generator)


    print(img_generator.generate_images(13056)[0].shape)
    print(img_generator)
    print(img_generator.generate_images(1088)[0].shape)
    print(img_generator)
    print(img_generator.generate_images(1200)[0].shape)
    print(img_generator)



