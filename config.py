#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

global global_config

class CommonConfig(object):
    tmp_dir = 'tmp/'
    tools_tf2torch_dir = 'toolbox/tf2torch/'
    tools_torch_dir = 'model/tools/'



class ToolConfig(object):
    batch_size=32
    epochs = 40
    lr_begin=1e-3
    lr_decay=0.97
    log_period = 20
    visual_cnt = 10
    save_period  = 50
    save_epoch_start = 3
    tools_num = 12

class RestoreConfig(object):
    screen_width = 63
    screen_height = 63
    screen_channel = 3
    batch_size = 16
    test_batch_size = 32
    stop_step = 3
    epoch = 10
    del_smooth = True

    max_step = 100000
    gamma = 0.95

    # img feature extract
    img_fm_size = 24*4*4

    # LSTM
    lstm_in = 32
    lstm_hidden = 50
    feature_map_size = 64

    # actor
    actor_hidden_size = 32


    # logging and test
    log_period = 100
    save_period = 2000

    visual_cnt = 30

    # tools fine-tune
    tools_ft_period = 20
    tools_ft_lr_decay_period = 400
    tools_ft_lr_decay = 0.99

class ExpConfig(CommonConfig):
    tool = ToolConfig
    RestoreConfig = RestoreConfig

def get_config(config1):
    config = ExpConfig
    for key, item in config1.__dict__.items():
        setattr(
            config, key, item)
    global global_config
    global_config = config
    return config
