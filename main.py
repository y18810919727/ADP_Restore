#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json
from config import get_config

import torch



import torch
import os
from toolbox.tool_train_main import tool_train, tools_test, load_tools_tf2torch, load_tools_torch
from adp.img_generator import ImageGenerator
import util

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def main(config):

    if 'ft' in config.event_identification:
        from adp.restore_ft import ImageRestoreFt as ImageRestore
    else:
        from adp.restore import ImageRestore

    print('Restorer Class:', ImageRestore.__name__)
    if config.tool_train_mode is 1:
        tool_train(config)
        return
    else:
        if config.tool_type == 'tf2torch':
            tools = load_tools_tf2torch(config.tools_tf2torch_dir, config)
        else:
            tools = load_tools_torch(config.tools_torch_dir, config)

    # test tools and terminate
    if config.tool_train_mode is 2:
        tools_test(tools, config)
        return

    restorer = ImageRestore(tools, config)
    if config.train_mode == 0 or config.train_mode == 2:
        train_img_generator = ImageGenerator(train_dir=config.train_dir)
        validation_img_generator = ImageGenerator(train_dir=config.validation_dir, shuffle=False)
        restorer.train_mode_set()
        restorer.train(train_img_generator, validation_img_generator)
    elif config.test_dataset in ['mild', 'moderate', 'severe', 'all']:
        def test_model_func():
            print('Data set: ',config.test_dataset)
            restorer.eval_mode_set()
            data_in = util.pngs_dir_read(os.path.join(
                config.test_dir, config.test_dataset+'_in')
            )
            data_gt = util.pngs_dir_read(os.path.join(
                config.test_dir, config.test_dataset+'_gt')
            )
            names = sorted([name for name in os.listdir(os.path.join(
                config.test_dir, config.test_dataset+'_in')
            )])
            names = list(
                map(
                    lambda x: x[:-7],
                    names
                )
            )
            result_dir = os.path.join(config.result_dir, config.test_dataset) if config.is_save else None
            if result_dir is not None and not os.path.exists(result_dir):
                os.makedirs(result_dir)
            restorer.test(data_in, data_gt, names, result_dir)
        if config.test_dataset == 'all':
            for ds_name in ['mild', 'moderate', 'severe']:
                config.test_dataset = ds_name
                test_model_func()
        else:
            test_model_func()

    else:
        # todo
        pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ADP-Restore')
    # 0 : nothing, 1: train tools, 2: test tools
    parser.add_argument('--tool_train_mode', type=int, default=0)
    parser.add_argument('--tool_train_data_dir', type=str, default='data/tool_data/train')
    parser.add_argument('--tool_validation_data_dir', type=str, default='data/tool_data/validation')
    parser.add_argument('--tool_train_test_dir', type=str, default='data/tool_data/test')
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--validation_dir', type=str, default='data/validation')
    parser.add_argument('--test_dir', type=str, default='data/test')
    parser.add_argument('--test_dataset', type=str, default='moderate',
                        help='select a dataset from mild or moderate or severe or all')

    parser.add_argument('--result_dir', type=str, default='data/result')
    parser.add_argument('--is_save', type=bool, default=False)

    parser.add_argument('--restorer_save_dir', type=str, default='model/restorer')
    parser.add_argument('--event_identification', type=str, default='')
    parser.add_argument('--restore_data_index', type=bool, default=True)

    # 0: retrain, 1: test 2: train based on the last checkpoint
    parser.add_argument('--train_mode', type=int, default=0)
    #  tf2torch, 1: torch(training by ourself)
    parser.add_argument('--tool_type', type=str, default='torch', help='tf2torch or torch')

    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--gpu_id', type=list, default=[0, 1, 2, 3], help='Available gpu id')
    config = parser.parse_args()
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    config.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    config = get_config(config)

    main(config)



