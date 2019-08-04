#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
import torch

from adp.restore import ImageRestore
from torch import nn
import util


class ImageRestoreNrl(ImageRestore):

    def __init__(self, tools, config):

        """
        naive ： original version
        nl ： add noise layer in policy net
        na ： No attention

        """
        from adp.actor_critic import Critic
        from adp.actor_critic import ActorNrl as Actor
        print('Policy_class: ', Actor.__name__)
        print('Critic_class: ', Critic.__name__)
        self.config = config
        self.train_mode = config.train_mode
        self.stop_step = config.RestoreConfig.stop_step
        self.tools = tools
        self.episode = {}
        self.imgs_gt = None
        self.event_identification = 'ADP'+config.event_identification
        self.actor = Actor(config)
        self.critic = Critic(config)
        self.max_step = config.RestoreConfig.max_step
        self.train_batch_size = config.RestoreConfig.batch_size
        self.train_generator_index = (0, 0)
        self.info_in_train = {
            'critic_loss': 0,
            'actor_loss': 0,
            'reward_sum': 0,
            'actions': np.zeros((self.stop_step, self.config.tool.tools_num))
        }

        # region ckpt 保存的东西

        self.step = 0
        self.actor_opt = torch.optim.Adam(self.actor.parameters())
        self.critic_opt = torch.optim.Adam(self.critic.parameters())
        self.critic_loss_func = torch.nn.MSELoss()
        self.max_reward_sum = -np.Inf
        self.event_time = util.cur_time_str()

        # endregion

        if self.train_mode >= 1:
            self.load()

        self.critic.to(config.device)
        self.actor.to(config.device)

        for tool_index in range(self.config.tool.tools_num):
            self.tools[tool_index].to(config.device)
            if self.config.device.type == 'cuda':
                self.tools[tool_index] = nn.DataParallel(self.tools[tool_index],
                                                         device_ids=self.config.gpu_id)

    def update_model(self, states, actions, rewards, values, imgs):
        """
        update critic network and policy network by using HDP algorithm
        Prokhorov, D. V., & Wunsch, D. C. (1997). Adaptive critic designs. IEEE Transactions on Neural Networks, 8(5), 997–1007. https://doi.org/10.1109/72.623201

        :param states:
        :param rewards:
        :param values:
        :param imgs:
        :return:
        """

        # region actor update
        actor_loss = self.cal_actor_loss2(actions, imgs)
        self.actor_opt.zero_grad()
        actor_loss.backward()  # The loss of critic net will backward later.
        self.actor_opt.step()
        #endregion

        return torch.zeros((1,1)).mean(), actor_loss

    def cal_actor_loss2(self, actions, imgs):
        """

        :param states:
        :param actions: bs * step * tool_num
        :param rewards:
        :param values:
        :return:
        """
        l2 = (imgs-self.imgs_gt)**2

        entropy = torch.distributions.Categorical(
            torch.softmax(actions, dim=2)
        ).entropy()
        if 'en' not in self.config.event_identification:
            entropy = entropy * 0
        return 10*l2.mean() - entropy.mean()/1000
