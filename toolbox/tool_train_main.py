#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
from tqdm import tqdm
from torch import nn
import torchvision.utils as vutils
import json
from util import h5extract
from util import MyDataset
import util

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from toolbox.tool_net import Net3, Net8
import time

def net_train(net, train_data, validation_data,config, tool_id):
    data = train_data['data']
    label = train_data['label']
    data_set = MyDataset(data, label)
    loader = DataLoader(
        dataset=data_set,
        batch_size=config.tool.batch_size,
        shuffle=True,
        num_workers=8,
    )
    opt = torch.optim.Adam(net.parameters(),lr=config.tool.lr_begin)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=config.tool.lr_decay)
    critic = torch.nn.MSELoss()
    event_id = util.cur_time_str()
    writer = SummaryWriter(logdir='./logs/tool/%02i/'%tool_id+event_id+'/')
    print('Tool: %02i ' % tool_id, 'Event id:', event_id)
    train_loss_in_period = 0
    with tqdm(total=len(loader)*config.tool.epochs) as pbar:
        min_test_loss = np.Inf
        for epoch in range(config.tool.epochs):
            scheduler.step()
            last_step = 0
            for step, (batch_x, batch_y) in enumerate(loader):
                batch_x = batch_x.to(config.device)
                batch_y = batch_y.to(config.device)
                y_estimate = net(batch_x)
                loss = critic(y_estimate+batch_x, batch_y)

                train_loss_in_period += float(loss.cpu())

                opt.zero_grad()
                loss.backward()
                opt.step()

                # region test , save and log
                if step % config.tool.log_period == 0 or (step + 1) == len(loader):
                    pbar.update(step-last_step)

                    test_x, test_y, test_y_estimate, test_loss, test_psnr, base_psnr = test_net(net, validation_data,
                                                                                                critic, config)

                    writer.add_scalar('data/test_loss', test_loss, step + epoch * len(loader))
                    writer.add_scalar('data/test_psnr', test_psnr, step + epoch * len(loader))
                    writer.add_scalar('data/base_psnr', float(int(base_psnr)), step + epoch * len(loader))
                    writer.add_scalar('data/learning_rate', opt.param_groups[0]['lr'] , step + epoch * len(loader))
                    if step - last_step > 0:
                        writer.add_scalar('data/train_loss', train_loss_in_period/(step - last_step),
                                          step + epoch * len(loader))
                        train_loss_in_period = 0

                    if step == 0 or (step + 1 == len(loader) and epoch + 1 == config.tool.epochs):

                        cnt = config.tool.visual_cnt
                        # visual_img = torch.cat(
                        #     (test_x[0:cnt], test_y_estimate[0:cnt], test_y[0:cnt]),
                        #     dim=0
                        # )
                        # visual_img = vutils.make_grid(visual_img, nrow=cnt)
                        # writer.add_image('img/validation', visual_img, step + epoch*len(loader))
                        # writer.add_images('img/validation_in', test_x[:cnt])
                        # writer.add_images('img/validation_out', test_y_estimate[:cnt])
                        # writer.add_images('img/validation_gt', test_y[:cnt])
                        for img_id in range(cnt):
                            writer.add_image('img/%02iin' % img_id, test_x[img_id],
                                             step + epoch*len(loader))
                            writer.add_image('img/%02iout' % img_id, test_y_estimate[img_id],
                                             step + epoch*len(loader))
                            writer.add_image('img/%02igt' % img_id, test_y[img_id],
                                             step + epoch*len(loader))

                        if epoch >= config.tool.save_epoch_start and test_loss < min_test_loss:
                            min_test_loss = min(min_test_loss, test_loss)
                            torch.save(net, os.path.join(config.tool_save_dir, 'tool%02i.pkd'%tool_id))
                    last_step = step

                # endregion


def test_net(net, validation_data, critic, config):
    test_x = validation_data['data']
    test_y = validation_data['label']
    test_x = torch.FloatTensor(test_x).to(config.device)
    test_y = torch.FloatTensor(test_y).to(config.device)
    test_y_estimate = net(test_x) + test_x
    test_loss = float(critic(test_y_estimate, test_y).cpu())
    test_psnr = util.psnr_cal(test_y_estimate.cpu().detach().numpy(),
                              test_y.cpu().detach().numpy())
    base_psnr = util.psnr_cal(test_x.cpu().detach().numpy(),
                              test_y.cpu().detach().numpy())

    return test_x, test_y, test_y_estimate, test_loss, test_psnr, base_psnr


def tool_train(config):
    for tool_id in range(6,8):
        # 4 tools in one kind of noisy group.  0, 1 tools for mild noise, 2, 3 tools for the severe.
        net = Net3() if tool_id % 4 < 2 else Net8()
        train_data = h5extract(
            os.path.join(
             config.tool_train_data_dir , str(tool_id)+'train.h5'
            )
        )

        validation_data = h5extract(
            os.path.join(
                config.tool_validation_data_dir ,str(tool_id)+'train.h5'
            )
        )
        net.to(config.device)
        #net = nn.DataParallel(net, device_ids=[0, 1, 2])
        net_train(net, train_data, validation_data, config, tool_id)


def tools_test(tools, config):

    for tool_id, tool in enumerate(tools):

        event_id = util.cur_time_str()
        if 'tool_data' in config.tool_validation_data_dir:
            test_data = h5extract(
                os.path.join(
                    config.tool_validation_data_dir, str(tool_id)+'train.h5'
                )
            )
            writer_name = './logs/tool/torch_test%02i/'%tool_id+event_id+'/'

        else:
            test_data = h5extract(
                os.path.join(
                    config.tool_validation_data_dir, 'validation.h5'
                )
            )
            writer_name = './logs/tool/torch_test_mix%02i/'%tool_id+event_id+'/'
        tool.to(config.device)

        writer = SummaryWriter(logdir=writer_name)

        test_x, test_y, test_y_estimate, test_loss, test_psnr, base_psnr = test_net(tool, test_data,
                                                                                    torch.nn.MSELoss(),
                                                                                    config)

        for write_iter in range(50000):
            writer.add_scalar('data/test_loss', test_loss, write_iter)
            writer.add_scalar('data/test_psnr', test_psnr, write_iter)
            writer.add_scalar('data/base_psnr', float(base_psnr), write_iter)
            writer.add_scalar('data/psnr_increment', float(test_psnr) - float(base_psnr), write_iter)

        cnt = config.tool.visual_cnt
        # visual_img = torch.cat(
        #     (test_x[0:cnt], test_y_estimate[0:cnt], test_y[0:cnt]),
        #     dim=0
        # )
        # visual_img = vutils.make_grid(visual_img, nrow=cnt)
        # writer.add_image('img/validation', visual_img, step + epoch*len(loader))
        # writer.add_images('img/validation_in', test_x[:cnt])
        # writer.add_images('img/validation_out', test_y_estimate[:cnt])
        # writer.add_images('img/validation_gt', test_y[:cnt])
        for img_id in range(cnt):
            writer.add_image('img/%02iin' % img_id, test_x[img_id],
                             0)
            writer.add_image('img/%02iout' % img_id, test_y_estimate[img_id],
                             0)
            writer.add_image('img/%02igt' % img_id, test_y[img_id],
                             0)
        print('Tool: %02i, psnr: %.02f, psnr increment: %.02f' % (
            tool_id, test_psnr, float(test_psnr) - float(base_psnr)))


