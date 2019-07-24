#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json
from config import get_config
from adp.restore import ImageRestore

import torch



import torch
import os
from toolbox.tool_train_main import tool_train, tools_test
from adp.img_generator import ImageGenerator
import util

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def main(config):
    if config.tool_train_mode is 1:
        tool_train(config)
        return
    elif config.tool_train_mode is 2:
        tools = util.load_tools(config.tool_save_dir, config.device)
        tools_test(tools, config)
        return
    else:
        tools = util.load_tools(config.tool_save_dir, config.device)
    restorer = ImageRestore(tools, config)
    if config.train_mode == 0 or config.train_mode == 2:
        train_img_generator = ImageGenerator(train_dir=config.train_dir)
        validation_img_generator = ImageGenerator(train_dir=config.validation_dir, shuffle=False)
        restorer.train(train_img_generator, validation_img_generator)
    else:
        # TODO test restorer
        pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ADP-Restore')
    # 0 : nothing, 1: train tools, 2: test tools
    parser.add_argument('--tool_train_mode', type=int, default=0)
    parser.add_argument('--tool_train_data_dir', type=str, default='data/tool_data/train')
    parser.add_argument('--tool_validation_data_dir', type=str, default='data/tool_data/validation')
    parser.add_argument('--tool_train_test_dir', type=str, default='data/tool_data/test')
    parser.add_argument('--tool_save_dir', type=str, default='model/tools')
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--validation_dir', type=str, default='data/validation')
    parser.add_argument('--test_dir', type=str, default='data/test')

    parser.add_argument('--restorer_save_dir', type=str, default='model/restorer')
    parser.add_argument('--event_identification', type=str, default='')
    parser.add_argument('--restore_data_index', type=bool, default=True)

    # 0: retrain, 1: test 2: train based on the last checkpoint
    parser.add_argument('--train_mode', type=int, default=0)


    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    config = parser.parse_args()
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    config.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    config = get_config(config)

    main(config)



