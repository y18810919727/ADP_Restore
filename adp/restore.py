#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import cv2
import os
import json
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch import  nn

import util
import torch
from tensorboardX import SummaryWriter

from adp.fusion import fuse_imgs


class ImageRestore(object):
    def __init__(self, tools, config):

        """
        naive ： original version
        nl ： add noise layer in policy net
        na ： No attention

        """
        from adp.actor_critic import Actor, Critic

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

    def train(self, train_img_generator, validation_img_generator):

        writer = SummaryWriter(logdir='./logs/RL/%s/%s/' % (self.event_identification, self.event_time))
        if self.config.restore_data_index:
            train_img_generator.restore_state(*self.train_generator_index)
        for self.step in tqdm(range(self.step, self.max_step)):  # self.step != 0 when training old model
            # del_smooth means delete the imgs which have psnr > 50 in training set.
            imgs_in, imgs_gt = train_img_generator.generate_images(self.train_batch_size,
                                                                   self.config.RestoreConfig.del_smooth)
            self.imgs_gt = torch.FloatTensor(imgs_gt).to(self.config.device)
            imgs = self.restore_train(imgs_in)
            states = self.episode['state']
            rewards = self.episode['reward']
            values = self.episode['values']
            actions = self.episode['actions']

            rewards = torch.stack(rewards, dim=1)  # (batch_size * stop_step)
            values = torch.stack(values, dim=1).squeeze(dim=-1)  # (batch_size * stop_step)
            states = torch.stack(states, dim=0)  # (stop_step+1) *  batch_size *  C*H*W
            actions = torch.stack(actions, dim=1)  # batch_size * stop_step * tool_num

            critic_loss, actor_loss = self.update_model(states, actions, rewards, values, imgs)

            self.info_in_train['critic_loss'] += float(critic_loss.detach().cpu())
            self.info_in_train['actor_loss'] += float(actor_loss.detach().cpu())
            self.info_in_train['reward_sum'] += float(rewards.detach().cpu().sum(dim=1).mean(dim=0))
            self.info_in_train['actions'] += actions.mean(dim=0).detach().cpu().numpy()
            #if self.step % self.config.RestoreConfig.log_period == self.config.RestoreConfig.log_period - 1 :
            if self.step % self.config.RestoreConfig.log_period == 0:
                self.train_generator_index = (train_img_generator.file_index,
                                              train_img_generator.data_index)
                self.model_validation(writer, validation_img_generator)
                self.info_in_train = {
                    'critic_loss': 0,
                    'actor_loss': 0,
                    'reward_sum': 0,
                    'actions': np.zeros((self.stop_step, self.config.tool.tools_num))
                }

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
        actor_loss = -1 * self.cal_actor_loss(states, actions, rewards, values)
        self.actor_opt.zero_grad()
        actor_loss.backward(retain_graph=True)  # The loss of critic net will backward later.
        self.actor_opt.step()
        #endregion

        # region critic update
        critic_loss = self.cal_critic_loss(states, rewards, values)
        self.critic_opt.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_opt.step()
        # endregion
        return critic_loss, actor_loss

    def cal_actor_loss(self, states, actions, rewards, values):
        """

        :param states:
        :param actions: bs * step * tool_num
        :param rewards:
        :param values:
        :return:
        """
        next_values = torch.cat(
            [values[:, 1:],
             torch.zeros(self.inference_batch_size, 1).to(self.config.device)],
            dim=1
        )
        entropy = torch.distributions.Categorical(
            torch.softmax(actions, dim=2)
        ).entropy() if 'en' in self.config.event_identification else 0
        return (rewards + self.config.RestoreConfig.gamma * next_values - entropy*0.1).mean()

    def cal_critic_loss(self, states, rewards, values):
        target_values = values[:, 1:].detach().clone()
        target_values = torch.cat(
            [target_values,
             torch.zeros(self.inference_batch_size, 1).to(self.config.device)],
            dim=1
        )
        target_values = rewards.detach() + self.config.RestoreConfig.gamma * target_values
        critic_loss = ((values - target_values)**2).mean()
        return critic_loss

    def restore_train(self, imgs_input):
        imgs = torch.FloatTensor(imgs_input).to(self.config.device)
        self.episode = {
            'state': [],
            'reward': [],
            'values': [],
            'actions': []
        }
        self.episode['state'].append(imgs)

        hidden_state_actor = None, None
        hidden_state_value = None, None
        self.inference_batch_size = imgs_input.shape[0]
        last_action = torch.zeros(self.inference_batch_size,
                                  self.config.tool.tools_num
                                  ).to(self.config.device)

        for step in range(self.stop_step):

            attention, hidden_state_actor = self.actor(imgs, hidden_state_actor, last_action)
            values, hidden_state_value = self.critic(imgs, hidden_state_value, last_action)

            last_action = attention
            out_imgs = fuse_imgs(imgs, attention, self.tools, self.config.device)
            self.episode['state'].append(out_imgs)
            self.episode['values'].append(values)
            self.episode['actions'].append(attention)
            self.episode['reward'].append(
                self.call_psnr(out_imgs) - self.call_psnr(imgs)
            )
            imgs = out_imgs
        # self.episode['values'].append(self.critic(imgs, hidden_state_value, last_action)[0])
        return imgs

    def call_psnr(self, img, img_gt=None):
        if img_gt is None:
            img_gt = self.imgs_gt
        if img_gt is None:
            raise ValueError('No ground truth input!')
        loss = (img - img_gt) ** 2
        eps = 1e-10
        loss_value = loss.mean(dim=(1,2,3)) + eps
        psnr = 10 * torch.log10(1.0 / loss_value)
        return psnr

    def model_validation(self, writer, validation_generator):
        sum_reward = 0
        rewards_steps = np.zeros(self.stop_step)
        critic_values_steps = np.zeros(self.stop_step)
        avg_critic_loss = 0
        test_actions = np.zeros((self.stop_step, self.config.tool.tools_num))
        imgs_in_cpu = []
        imgs_gt_cpu = []
        imgs_out_cpu = [[] for _ in range(self.stop_step)]
        validation_generator.restore_state(0, 0)
        for batch_id in range(validation_generator.data_len//self.config.RestoreConfig.test_batch_size):
            imgs_in, imgs_gt = validation_generator.generate_images(self.config.RestoreConfig.test_batch_size)
            self.imgs_gt = torch.FloatTensor(imgs_gt).to(self.config.device)
            imgs_out = self.restore_train(imgs_in)

            states = self.episode['state']
            rewards = self.episode['reward']
            values = self.episode['values']
            actions = self.episode['actions']


            imgs_in_cpu.append(imgs_in)
            imgs_gt_cpu.append(imgs_gt)
            for step_index in range(self.stop_step):
                imgs_out_cpu[step_index].append(states[step_index+1].detach().cpu().numpy())

            rewards = torch.stack(rewards, dim=1)  # (batch_size * stop_step)
            values = torch.stack(values, dim=1).squeeze(dim=-1)  # (batch_size * stop_step)

            actions = torch.stack(actions, dim=1)  # batch_size * stop_step * tool_num

            critic_loss = self.cal_critic_loss(states, rewards, values)
            avg_critic_loss += float(critic_loss.cpu().detach())

            # record average critic values
            critic_values_steps += values.mean(dim=0).detach().cpu().numpy()

            # record rewards gained in each process step
            rewards_steps += rewards.mean(dim=0).detach().cpu().numpy()

            # record summary rewards gained
            sum_reward += rewards.mean(dim=0).detach().cpu().numpy().sum()

            test_actions += actions.mean(dim=0).detach().cpu().numpy()

        cnt = self.config.RestoreConfig.visual_cnt

        imgs_in_cpu = np.concatenate(imgs_in_cpu)
        imgs_gt_cpu = np.concatenate(imgs_gt_cpu)

        avg_critic_loss = avg_critic_loss / (validation_generator.data_len//self.config.RestoreConfig.test_batch_size)
        rewards_steps = rewards_steps / (validation_generator.data_len//self.config.RestoreConfig.test_batch_size)
        critic_values_steps = critic_values_steps / (validation_generator.data_len//self.config.RestoreConfig.test_batch_size)
        sum_reward = sum_reward / (validation_generator.data_len//self.config.RestoreConfig.test_batch_size)
        test_actions = test_actions / (validation_generator.data_len//self.config.RestoreConfig.test_batch_size)


        psnr_final = float(self.call_psnr(imgs_out).mean().detach().cpu())

        writer.add_scalar('data/test_critic_loss', avg_critic_loss, self.step)
        writer.add_scalar('data/test_sum_reward', sum_reward, self.step)
        writer.add_scalar('data/test_psnr_final', psnr_final, self.step)
        writer.add_scalar('data/train_critic_loss',
                          self.info_in_train['critic_loss']/self.config.RestoreConfig.log_period, self.step)
        writer.add_scalar('data/train_actor_loss',
                          self.info_in_train['actor_loss']/self.config.RestoreConfig.log_period, self.step)
        writer.add_scalar('data/train_reward_sum',
                          self.info_in_train['reward_sum']/self.config.RestoreConfig.log_period, self.step)

        train_attention = self.info_in_train['actions'] / self.config.RestoreConfig.log_period
        for step_index in range(self.stop_step):
            plt.figure()
            plt.bar(range(1, self.config.tool.tools_num+1), train_attention[step_index])
            writer.add_figure('figure/train_action%i' % step_index, plt.gcf(), self.step)
            plt.figure()
            plt.bar(range(1, self.config.tool.tools_num+1), test_actions[step_index])
            writer.add_figure('figure/test_action%i' % step_index, plt.gcf(), self.step)

        for step_index in range(self.stop_step):
            writer.add_scalar('data/test_reward_step%02i' % step_index, rewards_steps[step_index], self.step)
            writer.add_scalar('data/test_critic_step%02i' % step_index, critic_values_steps[step_index], self.step)

        for step_index in range(self.stop_step):
            imgs_out_cpu[step_index] = np.concatenate(imgs_out_cpu[step_index], axis=0)

        for img_id in range(cnt):
            writer.add_image('img/%02iin' % img_id, imgs_in_cpu[img_id],
                             self.step)
            writer.add_image('img/%02igt' % img_id, imgs_gt_cpu[img_id],
                             self.step)

            for step_index in range(self.stop_step):
                writer.add_image('img/%02iout%02i' % (img_id, step_index), imgs_out_cpu[step_index][img_id],
                                 self.step)
        if self.step % self.config.RestoreConfig.save_period == 0 and sum_reward > self.max_reward_sum:
            self.max_reward_sum = sum_reward
            self.save()

    # 目前还没用到
    def restore_test(self, imgs_input):
        imgs = torch.FloatTensor(imgs_input).to(self.config.device)
        self.episode = {
            'state': [],
            'reward': [],
            'values': []
        }
        hidden_state_actor = None, None
        hidden_state_value = None, None

        self.inference_batch_size = imgs_input.shape[0]
        last_action = torch.zeros(self.inference_batch_size,
                                  self.config.tool.tools_num
                                  ).to(self.config.device)
        for step in range(self.stop_step):

            attention, hidden_state_actor = self.actor(imgs, hidden_state_actor, last_action)
            values, hidden_state_value = self.critic(imgs, hidden_state_actor, last_action)

            last_action = attention
            out_imgs = fuse_imgs(imgs, attention, self.tools, self.config.device)
            self.episode['reward'].append(
                self.call_psnr(out_imgs) - self.call_psnr(imgs)
            )
            imgs = out_imgs
        # self.episode['values'].append(self.critic(imgs, hidden_state_value, last_action)[0])
        return imgs

    def train_mode_set(self):
        self.actor.train()
        self.critic.train()
        for tool in self.tools:
            tool.train()

    def eval_mode_set(self):
        self.actor.eval()
        self.critic.eval()
        for tool in self.tools:
            tool.eval()

    def save(self):
        state = {
            'critic': self.critic.state_dict(),
            'actor': self.actor.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'critic_loss_func': self.critic_loss_func.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'step': self.step,
            'critic_loss': self.info_in_train['critic_loss'],
            'actor_loss': self.info_in_train['actor_loss'],
            'reward_sum': self.info_in_train['reward_sum'],
            'actions': self.info_in_train['actions'],
            'train_generator_index': self.train_generator_index,
            'max_reward_sum': self.max_reward_sum,
            'event_time': self.event_time,
        }
        for tool_index in range(self.config.tool.tools_num):
            tool = self.tools[tool_index]
            state['tool%i' % tool_index] = tool.module.state_dict() if type(tool) is torch.nn.DataParallel \
                else tool.state_dict()
        torch.save(state, os.path.join(self.config.restorer_save_dir, self.event_identification+'.pth'))

    def load(self):
        ckpt = torch.load(os.path.join(self.config.restorer_save_dir, self.event_identification+'.pth'))
        self.critic.load_state_dict(ckpt['critic'])
        self.actor.load_state_dict(ckpt['actor'])
        self.critic_opt.load_state_dict(ckpt['critic_opt'])
        self.actor_opt.load_state_dict(ckpt['actor_opt'])
        self.step = ckpt['step'] + 1
        self.event_time = ckpt['event_time'] if 'event_time' in ckpt.keys() else self.event_time
        self.info_in_train['critic_loss'] = ckpt['critic_loss']
        self.info_in_train['reward_sum'] = ckpt['reward_sum']
        self.info_in_train['actions'] = ckpt['actions'] if 'actions' in ckpt.keys() \
            else np.zeros((self.stop_step, self.config.tool.tools_num))
        self.info_in_train['actor_loss'] = ckpt['actor_loss']
        self.critic_loss_func = ckpt['critic_loss_func'] if 'critic_loss_func' in ckpt.keys() \
            else torch.nn.MSELoss()
        self.train_generator_index = ckpt['train_generator_index']
        self.max_reward_sum = ckpt['max_reward_sum']

        if 'tool%i' % 0 in ckpt.keys():
            for tool_index in range(self.config.tool.tools_num):
                self.tools[tool_index].load_state_dict(ckpt['tool%i' % tool_index])

    def test(self, data_in, data_gt, names, result_dir):
        base_psnr = util.psnr_cal(data_in, data_gt)
        data_len = data_in.shape[0]
        imgs_in_cpu = []
        imgs_gt_cpu = []
        imgs_out_cpu = [[] for _ in range(self.stop_step)]
        rewards_steps = np.zeros(self.stop_step)
        psnr_step = np.zeros(self.stop_step)
        sum_reward = 0
        psnr_final = 0
        test_actions = np.zeros((self.stop_step, self.config.tool.tools_num))

        for batch_id in range(data_len//self.config.RestoreConfig.test_batch_size):
            beg = batch_id * self.config.RestoreConfig.test_batch_size
            end = batch_id * self.config.RestoreConfig.test_batch_size + self.config.RestoreConfig.test_batch_size
            imgs_in = data_in[beg:end, ...]
            imgs_gt = data_gt[beg:end, ...]
            self.imgs_gt = torch.FloatTensor(imgs_gt).to(self.config.device)
            imgs_out = self.restore_train(imgs_in)

            states = self.episode['state']
            rewards = self.episode['reward']
            values = self.episode['values']
            actions = self.episode['actions']


            imgs_in_cpu.append(imgs_in)
            imgs_gt_cpu.append(imgs_gt)
            for step_index in range(self.stop_step):
                imgs_out_cpu[step_index].append(states[step_index+1].detach().cpu().numpy())
                psnr_step[step_index] += float(self.call_psnr(states[step_index+1]).detach().mean().cpu())


            rewards = torch.stack(rewards, dim=1)  # (batch_size * stop_step)
            values = torch.stack(values, dim=1).squeeze(dim=-1)  # (batch_size * stop_step)
            actions = torch.stack(actions, dim=1)  # batch_size * stop_step * tool_num


            # record rewards gained in each process step
            rewards_steps += rewards.mean(dim=0).detach().cpu().numpy()

            # record psnr in each process step

            # record summary rewards gained
            sum_reward += rewards.mean(dim=0).detach().cpu().numpy().sum()

            test_actions += actions.mean(dim=0).detach().cpu().numpy()

            psnr_final += float(self.call_psnr(imgs_out).mean().detach().cpu())

        imgs_in_cpu = np.concatenate(imgs_in_cpu)
        imgs_gt_cpu = np.concatenate(imgs_gt_cpu)
        for step_index in range(self.stop_step):
            imgs_out_cpu[step_index] = np.concatenate(imgs_out_cpu[step_index])
        rewards_steps = rewards_steps / (data_len//self.config.RestoreConfig.test_batch_size)
        sum_reward = sum_reward / (data_len//self.config.RestoreConfig.test_batch_size)
        psnr_step = psnr_step / (data_len//self.config.RestoreConfig.test_batch_size)
        psnr_final = psnr_final / (data_len//self.config.RestoreConfig.test_batch_size)
        test_actions = test_actions / (data_len//self.config.RestoreConfig.test_batch_size)

        if result_dir is not None:
            for img_id in range(data_len):

                cv2.imwrite(os.path.join(result_dir, names[img_id]+'_in.png'),
                            (imgs_in_cpu[img_id]*255).transpose(1,2,0))

                cv2.imwrite(os.path.join(result_dir, names[img_id]+'_gt.png'),
                            (imgs_gt_cpu[img_id]*255).transpose(1,2,0))
                for step_index in range(self.stop_step):

                    cv2.imwrite(os.path.join(result_dir, names[img_id]+'_out%i.png' % (step_index+1)),
                                (imgs_out_cpu[step_index][img_id]*255).transpose(1,2,0))

        print('Final result:')

        for m in range(self.stop_step):
            print('Weight %d:' % (m+1), test_actions[m])
        for m in range(self.stop_step):
            print(
                ('reward' + str(m + 1) + ': %.4f, psnr' + str(m + 1) + ': %.4f' +
                 ', total tested images: %d') % (rewards_steps[m], psnr_step[m], data_len)
                  )

        print('Base psnr: %.4f, Final psnr: %.4f, Sum reward: %.4f' %
              (base_psnr, psnr_final, sum_reward))



