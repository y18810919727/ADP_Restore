#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch



from adp.restore import ImageRestore

class ImageRestoreNl(ImageRestore):

    def update_model(self, states, rewards, values, imgs):

        critic_loss, actor_loss = super(ImageRestoreNl, self).update_model(states, rewards, values, imgs)

        if 'nl' in self.config.event_identification:
            self.actor.actor_linear1.reset_noise()
            self.actor.actor_linear2.reset_noise()

        return critic_loss, actor_loss

