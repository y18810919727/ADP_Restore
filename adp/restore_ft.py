#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
import torch

from adp.restore import ImageRestore


class ImageRestoreFt(ImageRestore):

    def __init__(self, tools, config):
        super(ImageRestoreFt, self).__init__(tools, config)
        self.tools_opt = []
        self.tools_scheduler = []
        self.tools_critic = torch.nn.MSELoss()
        for tool_id in range(config.tool.tools_num):
            self.tools_opt.append(torch.optim.Adam(self.tools[tool_id].parameters(), lr=1e-4))
            self.tools_scheduler.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.tools_opt[-1], gamma=config.RestoreConfig.tools_ft_lr_decay))

    def update_model(self, states, rewards, values, imgs):
        critic_loss, actor_loss = super(ImageRestoreFt, self).update_model(states, rewards, values, imgs)
        if not self.step % self.config.RestoreConfig.tools_ft_period == 0:
            return critic_loss, actor_loss
        tools_loss = self.tools_critic(imgs, self.imgs_gt)
        for tool_id in range(self.config.tool.tools_num):
            opt = self.tools_opt[tool_id]
            opt.zero_grad()
            for param in self.tools[tool_id].parameters():
                param.grad = torch.autograd.grad(tools_loss, param, retain_graph=True)[0]
            opt.step()
            if self.step % self.config.RestoreConfig.tools_ft_lr_decay_period == 0:
                self.tools_scheduler[tool_id].step()

        return critic_loss, actor_loss
