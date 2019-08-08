import numpy as np
import math
import cv2
import os
import json
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch import  nn

import util
import torch
from tensorboardX import SummaryWriter
from adp.fusion import fuse_imgs
from adp.restore import ImageRestore

class ImageRestoredqn(ImageRestore):

    def __init__(self, tools, config):
        super(ImageRestoredqn, self).__init__(tools, config)
        self.last_action = None

    def train(self, train_img_generator, validation_img_generator):
        writer = SummaryWriter(logdir='./logs/RL/%s/%s/' % (self.event_identification, self.event_time))
        if self.config.restore_data_index:
            train_img_generator.restore_state(*self.train_generator_index)
        for self.step in tqdm(range(self.step, self.max_step)):  # self.step != 0 when training old model
            # del_smooth means delete the imgs which have psnr > 50 in training set.
            imgs_in, imgs_gt = train_img_generator.generate_images(self.train_batch_size,
                                                                   self.config.RestoreConfig.del_smooth)
            self.imgs_gt = torch.FloatTensor(imgs_gt).to(self.config.device)
            imgs = self.restore_train(imgs_in)
            states = self.episode['state']
            rewards = self.episode['reward']
            values = self.episode['values']
            actions = self.episode['actions']



    def restore_train(self, imgs_input):
        imgs = torch.FloatTensor(imgs_input).to(self.config.device)
        self.episode = {
            'state': [],
            'reward': [],
            'values': [],
            'actions': []
        }
        self.episode['state'].append(imgs)

        hidden_state_actor = None, None
        hidden_state_value = None, None
        self.inference_batch_size = imgs_input.shape[0]
        if self.last_action == None:
            self.last_action = torch.zeros(self.inference_batch_size,
                                           self.config.tool.tools_num
                                  ).to(self.config.device)

        for step in range(self.stop_step):

            values, hidden_state_value = self.critic(imgs, hidden_state_value, self.last_action)
            v, action = torch.max(values,1)
            out_imgs = fuse_imgs(imgs, self.tools, self.config.device)
            self.episode['state'].append(out_imgs)
            self.episode['values'].append(values)
            self.episode['actions'].append(action)
            self.episode['reward'].append(
                self.call_psnr(out_imgs) - self.call_psnr(imgs)
            )
            imgs = out_imgs
        # self.episode['values'].append(self.critic(imgs, hidden_state_value, last_action)[0])
        return imgs







