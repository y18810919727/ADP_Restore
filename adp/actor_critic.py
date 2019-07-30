#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import torch

import json

import torch.nn as nn

from torch.nn import functional as F
from torch.nn import Module
from common.layers import NoisyLinear


class Critic(Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        self.config = config

        self.conv1 = nn.Conv2d(3, 32, 9, stride=2,padding=4)
        self.conv2 = nn.Conv2d(32, 24, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(24, 24, 5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(24, 24, 5, stride=2, padding=2)

        self.fm_linear = nn.Linear(config.RestoreConfig.img_fm_size,
                                   config.RestoreConfig.feature_map_size)

        self.critic_lstm = nn.LSTM(input_size=config.RestoreConfig.feature_map_size+config.tool.tools_num,
                                   hidden_size=config.RestoreConfig.lstm_hidden,
                                   )
        self.critic_linear = nn.Linear(config.RestoreConfig.lstm_hidden, 1)

    def feature_extract(self, imgs):

        out_imgs = self.conv4(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(imgs)))))))
        assert out_imgs.shape[1:] == (24, 4, 4, )
        fm = out_imgs.reshape(-1, self.config.RestoreConfig.img_fm_size)
        assert fm.shape[1:] == (self.config.RestoreConfig.img_fm_size, )
        return self.fm_linear(fm)

    def value_lstm(self, fm, hidden_state, last_action):

        fm = torch.cat([fm, last_action.detach()], dim=1)

        fm = fm.unsqueeze(0)
        h, c = hidden_state

        if h is None:
            out_fm, (h, c) = self.critic_lstm(fm)
        else:
            out_fm, (h, c) = self.critic_lstm(fm, (h, c))

        out_fm = out_fm.squeeze(dim=0)
        value = self.critic_linear(out_fm)
        return value, (h, c)

    def forward(self, imgs, hidden_state, last_action):
        fm = self.feature_extract(imgs)

        return self.value_lstm(fm, hidden_state, last_action=last_action)


class Actor(Module):

    def __init__(self, config):
        super(Actor, self).__init__()
        self.config = config

        self.conv1 = nn.Conv2d(3, 32, 9, stride=2,padding=4)
        self.conv2 = nn.Conv2d(32, 24, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(24, 24, 5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(24, 24, 5, stride=2, padding=2)

        self.fm_linear = nn.Linear(config.RestoreConfig.img_fm_size,
                                   config.RestoreConfig.feature_map_size)

        self.actor_lstm = nn.LSTM(input_size=config.RestoreConfig.feature_map_size+config.tool.tools_num,
                                  hidden_size=config.RestoreConfig.lstm_hidden
                                  )
        if self.is_symbol('na'):
            node_num = config.tool.tools_num + 1
        elif self.is_symbol('sa'):
            node_num = config.tool.tools_num + 3
        else:
            node_num = config.tool.tools_num
        if self.is_symbol('nl'):
            self.actor_linear1 = NoisyLinear(config.RestoreConfig.lstm_hidden, config.RestoreConfig.actor_hidden_size,
                                             config.device, std_init=0.3)
            self.actor_linear2 = NoisyLinear(config.RestoreConfig.actor_hidden_size, node_num,
                                             config.device, std_init=0.3)
        else:
            self.actor_linear1 = nn.Linear(config.RestoreConfig.lstm_hidden, config.RestoreConfig.actor_hidden_size)
            self.actor_linear2 = nn.Linear(config.RestoreConfig.actor_hidden_size, node_num)

    def feature_extract(self, imgs):

        out_imgs = self.conv4(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(imgs)))))))
        assert out_imgs.shape[1:] == (24, 4, 4, )
        fm = out_imgs.reshape(-1, self.config.RestoreConfig.img_fm_size)
        assert fm.shape[1:] == (self.config.RestoreConfig.img_fm_size, )
        return self.fm_linear(fm)

    def policy_lstm(self, fm, hidden_state, last_action):
        h, c = hidden_state
        fm = torch.cat([fm, last_action.detach()], dim=1)
        fm = fm.unsqueeze(0)

        if h is None:
            out_fm, (h, c) = self.actor_lstm(fm)
        else:
            out_fm, (h, c) = self.actor_lstm(fm, (h, c))

        out_fm = out_fm.squeeze(dim=0)
        tools_weight = self.actor_linear2(
            F.relu(self.actor_linear1(out_fm))
        )
        if self.is_symbol('na'):
            attention = F.softmax(tools_weight[:, :self.config.tool.tools_num], dim=1)
            margin = torch.sigmoid(tools_weight[:, -1:])*2
            attention = attention * margin
        elif self.is_symbol('sa'):
            attention_blur = F.softmax(tools_weight[:, 0:4], dim=1) * torch.sigmoid(tools_weight[:,12:13])*2/3.0
            attention_gauss = F.softmax(tools_weight[:, 4:8], dim=1), * torch.sigmoid(tools_weight[:,13:14])*2/3.0
            attention_jpg = F.softmax(tools_weight[:, 8:12], dim=1) * torch.sigmoid(tools_weight[:,14:15])*2/3.0
            attention = torch.cat([attention_blur, attention_gauss, attention_jpg], dim=1)
        else:
            attention = F.softmax(tools_weight, dim=1)
        return attention, (h, c)

    def forward(self, imgs, hidden_state, last_action):
        fm = self.feature_extract(imgs)

        return self.policy_lstm(fm, hidden_state, last_action)

    def is_symbol(self, symbol):
        if symbol in self.config.event_identification:
            return True
        else:
            return False
