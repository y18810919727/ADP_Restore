#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch.nn import Module
from torch import nn
from torch.nn import functional as F


class Net3(Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(32, 16, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(16, 3, 5, stride=1, padding=2)

    def forward(self, input):
        return self.conv3(F.relu(self.conv2(F.relu(self.conv1(input)))))


class Net8(Module):
    def __init__(self):
        super(Net8, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 32, 1, stride=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(32, 64, 1, stride=1)
        self.conv8 = nn.Conv2d(64, 3, 5, stride=1, padding=2)

    def forward(self, input):
        self.fm1 = self.conv2(F.relu(self.conv1(input)))
        self.fm2 = self.conv4(F.relu(self.conv3(F.relu(self.fm1)))) + self.fm1
        self.fm3 = self.conv5(F.relu(self.conv6(F.relu(self.fm2)))) + self.fm2
        self.res = self.conv8(F.relu(self.conv7(self.fm3)))
        return self.res








