#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
from util import h5extract
import util
import os
import json

from tensorboardX import SummaryWriter

import torch
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]=''
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
gpu_options = tf.GPUOptions(allow_growth=True)

def psnr_cal(im_input, im_label):
    loss = (im_input - im_label) ** 2
    eps = 1e-10
    loss_value = loss.mean() + eps
    psnr = 10 * math.log10(1.0 / loss_value)
    return psnr

class TfBoxesTest(object):
    def __init__(self, toolbox_path):
        self.sessions = []
        self.graphs = []
        self.inputs = []
        self.outputs = []
        for idx in range(12):
            print(idx)
            g = tf.Graph()
            with g.as_default():
                # load graph for Graph g
                saver = tf.train.import_meta_graph(toolbox_path + 'tool%02d' % (idx + 1) + '.meta')
                # input data
                input_data = g.get_tensor_by_name('Placeholder:0')
                self.inputs.append(input_data)
                # get the output
                output_data = g.get_tensor_by_name('sum:0')
                self.outputs.append(output_data)
                # save graph
                self.graphs.append(g)
            sess = tf.Session(graph=g, config=tf.ConfigProto(log_device_placement=True,
                                                             gpu_options=gpu_options))
            with g.as_default():
                with sess.as_default():
                    saver.restore(sess, toolbox_path + 'tool%02d' % (idx + 1))
                    self.sessions.append(sess)

    def test_data(self, data_path):
        for index in range(12):
            # img shape N * bs * H * W * channels
            log_dir = './logs/'
            data = h5extract(os.path.join(data_path, '%itrain.h5' % index))
            imgs_in = data['data'].transpose(0,2,3,1)
            imgs_gt = data['label'].transpose(0,2,3,1)
            imgs_out = np.copy(imgs_in)
            psnr_out = np.zeros(imgs_gt.shape[0])
            psnr_increment = np.zeros(imgs_gt.shape[0])
            psnr_base = np.zeros(imgs_gt.shape[0])


            event_id = util.cur_time_str()
            writer = SummaryWriter(logdir='./logs/tool/tf_tool%02i/'%index+event_id+'/')

            for img_id in range(imgs_gt.shape[0]):
                img_in = imgs_in[img_id:img_id+1, ...]
                #img_out = imgs_out[img_id:img_id+1, ...]
                img_gt = imgs_gt[img_id:img_id+1, ...]
                feed_dict = {
                    self.inputs[index]: img_in
                }
                with self.graphs[index].as_default():
                    with self.sessions[index].as_default():
                        img_out = self.sessions[index].run(self.outputs[index], feed_dict=feed_dict)

                imgs_out[img_id] = img_out

                psnr_base[img_id] = psnr_cal(img_in, img_gt)
                psnr_out[img_id] = psnr_cal(img_out, img_gt)
                psnr_increment[img_id] = psnr_out[img_id] - psnr_base[img_id]
                if img_id < 10:
                    writer.add_image('img/%02iin' % img_id, img_in[0].transpose(2,0,1),
                                     0)
                    writer.add_image('img/%02iout' % img_id, img_out[0].transpose(2,0,1),
                                     0)
                    writer.add_image('img/%02igt' % img_id, img_gt[0].transpose(2,0,1),
                                     0)

            print('Tool: %02i, psnr: %.02f, psnr increment: %.02f' % (
                index, psnr_out.mean(), psnr_increment.mean()))

            # for index in range(12):
            #     writer = tf.summary.FileWriter(logdir=log_dir+'tf_tools%02d/' % index,
            #                                    session=self.sessions[index].graph)

            for write_ti in range(50000):
                writer.add_scalar('data/test_psnr', psnr_out.mean(), write_ti)
                writer.add_scalar('data/base_psnr', psnr_base.mean(), write_ti)
                writer.add_scalar('data/psnr_increment', psnr_increment.mean(), write_ti)


with tf.Session(config=tf.ConfigProto(log_device_placement=True,
                                      gpu_options=gpu_options)) as sess_all:
    #tools_path = os.path.join('')
    test_path = os.path.join('data', 'tool_data', 'validation')
    tf_box_test = TfBoxesTest(
        toolbox_path='/home1/jupyter_data/Yuanzhaolin/remote_host/ADP_Restore/toolbox/')
    tf_box_test.test_data(
        '/home1/jupyter_data/Yuanzhaolin/remote_host/ADP_Restore/data/tool_data/validation/')

