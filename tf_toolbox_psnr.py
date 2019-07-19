#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch

for idx in range(12):
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
    sess = tf.Session(graph=g, config=tf.ConfigProto(log_device_placement=True))
    with g.as_default():
        with sess.as_default():
            saver.restore(sess, toolbox_path + 'tool%02d' % (idx + 1))
            self.sessions.append(sess)
            writer = tf.summary.FileWriter(self.log_dir + '/'+'tool%02d' % (idx+1)+'/', sess.graph)
