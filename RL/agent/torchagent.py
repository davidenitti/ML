from numba import jit, jitclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import os, sys, shutil
import gym
import time
import numpy as np
from torch.autograd import Variable
import logging
from . import common
from . import models
from . import buffers

from .agent_utils import onehot, vis
import json

import matplotlib.pyplot as plt0

try:
    import IPython.display  # for ipython/notebook compatibility
except:
    plt0.ion()
logger = logging.getLogger(__name__)


class deepQconv(object):
    def save(self, filename=None):
        if filename is None:
            filename = self.config["path_exp"]
        with open(filename + ".json", "w") as input_file:
            json.dump(self.config, input_file, indent=3)
        if self.config['save_mem']:
            logger.info('saving memory')
            buffers.save_zipped_pickle(self.memory, filename + "_mem.p", zip=False)
        checkpoint = {"optimizer": self.optimizer.state_dict()}
        for m in self.models:
            checkpoint[m] = self.models[m].state_dict()
        torch.save(checkpoint, filename + ".pth")

    def __init__(self, observation_space, action_space, reward_range, userconfig):
        self.config = userconfig
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_range = reward_range

        self.isdiscrete = isinstance(self.action_space, gym.spaces.Discrete)
        print(self.action_space)
        print(self.config)
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise Exception('Observation space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(
                observation_space, self))
        self.learnrate = self.config['initial_learnrate']
        self.initQnetwork()

    def plot(self, w, lists, reward_threshold, plt, plot=True, numplot=0, start_episode=0):
        width = 15000
        totrewlist, test_rew_smooth, test_rew_epis = lists
        if len(totrewlist) > 5:
            if plot:
                fig = plt.figure(numplot)
                fig.canvas.set_window_title(str(self.config["path_exp"]) + " " + str(self.config))

            plt.clf()
            plt.xlim(start_episode + max(-1, len(totrewlist) / 50 * 50 - width),
                     start_episode + len(totrewlist) // 50 * 50 + 50)
            plt.ylim(min(totrewlist[max(0, len(totrewlist) // 50 * 50 - width):]) - 5,
                     max(totrewlist[max(0, len(totrewlist) // 50 * 50 - width):]) + 5)
            plt.plot([start_episode, start_episode + len(totrewlist) + 100], [reward_threshold, reward_threshold],
                     color='green')

            plt.plot(range(start_episode, start_episode + len(totrewlist)), totrewlist, color='red')
            # plt.plot(range(len(total_rew_discountlist)), [ddd-total_rew_discountlist[0]+totrewlist[0] for ddd in total_rew_discountlist], color='green')
            plt.plot(test_rew_epis[1], test_rew_smooth, color='cyan')

            plt.scatter(test_rew_epis[1], test_rew_epis[0], color='black')

            plt.plot([start_episode + max(0, len(totrewlist) - 1 - 20), start_episode + len(totrewlist) - 1],
                     [np.mean(np.array(test_rew_epis[0][-20:])), np.mean(np.array(test_rew_epis[0][-20:]))],
                     color='blue')

            plt.plot([start_episode + max(0, len(totrewlist) - 1 - 100), start_episode + len(totrewlist) - 1],
                     [np.mean(np.array(totrewlist[-100:])), np.mean(np.array(totrewlist[-100:]))], color='black')
            if self.config["path_exp"]:
                plt.savefig(self.config["path_exp"] + '_reward' + '.png', dpi=400)

            ymin, ymax = plt.gca().get_ylim()
            xmin, xmax = plt.gca().get_xlim()
            plt.text(xmin, ymax, str(self.config), ha='left', wrap=True, fontsize=18)
            if self.config['conv'] and False:  # fixme
                if plot:
                    fig = plt.figure(2)
                    fig.canvas.set_window_title(str(self.config["path_exp"]) + " " + str(self.config))

                if self.memory.sizemem() > 1 + self.config['past']:
                    selec = self.config['past'] + np.random.choice(self.memory.sizemem() - 1 - self.config['past'])
                    imageselected = self.memory[selec][0].copy()
                    filtered1 = np.zeros((1, 1, 10, 10))
                    filtered2 = np.zeros((1, 1, 10, 10))
                    vis(plt, w, imageselected, filtered1[0], filtered2[0])

                    if self.config["path_exp"] and plot == False:
                        plt.savefig(self.config["path_exp"] + '_features' + '.png', dpi=300)

            if 'transition_net' in self.config and self.config['transition_net']:
                if len(self.state_list[0]) > 0 and len(self.state_list[1]) > 0:
                    st = []
                    st.append(np.vstack(self.state_list[0])[:, :2].T)
                    st.append(np.vstack(self.state_list[1])[:, :2].T)
                    self.plot_state(state_list=st)

            if plot:
                plt.draw()
                plt.pause(.001)
                fig.canvas.flush_events()
        time.sleep(0.01)

    def scaleobs(self, obs):
        if np.isinf(self.observation_space.low).any() or np.isinf(self.observation_space.high).any():
            return obs
        else:
            o = (obs - self.observation_space.low) / (
                    self.observation_space.high - self.observation_space.low) * 2. - 1.
            return o

    def epsilon(self, episode=None):
        if episode == None:
            return 0.
        elif episode == -1:
            return self.config['testeps']
        else:
            if 'exp_decay' in self.config:
                assert 'linear_decay' not in self.config
                return max(self.config['mineps'], self.config['eps'] * self.config['exp_decay'] ** episode)
            else:
                return max(self.config['mineps'],
                           self.config['eps'] - (self.config['eps'] - self.config['mineps']) * self.config[
                               'linear_decay'] *
                           self.config['num_updates'])

    def getlearnrate(self):
        return self.learnrate

    def commoninit(self):
        chk, msg = common.checkparams(self.config)
        if chk == False:
            raise Exception(msg)

        self.fulldouble = self.config['doubleQ'] and self.config['copyQ'] <= 0
        self.copyQalone = self.config['copyQ'] > 0 and self.config['doubleQ'] == False

        self.useConv = self.config['conv']
        if 'dimdobs' in self.config:
            self.scaled_obs = self.config['dimdobs']
        else:
            self.scaled_obs = self.observation_space.shape

    def sharednet(self, input, state_dict=None):
        if self.useConv:
            dense_conv = models.ConvNet(self.config['convlayers'], input,
                                        activation=self.config['activation'],
                                        # numchannels=1 + 1 * self.config['past'],
                                        scale=self.config['scaleobs'],
                                        init_weight=self.config["init_weight"])
            dense_conv_shape = dense_conv(torch.zeros([1] + input)).shape
            if len(self.config['sharedlayers']) > 0:
                dense = models.DenseNet(self.config['sharedlayers'], input_shape=dense_conv_shape[-1],
                                        scale=1, final_act=True,
                                        activation=self.config['activation'],
                                        batch_norm=self.config["batch_norm"],
                                        init_weight=self.config["init_weight"])
                dense = nn.Sequential(dense_conv, dense)
                num_features = self.config['sharedlayers'][-1]
            else:
                dense = dense_conv
                num_features = dense_conv_shape[-1]
        else:
            if len(self.config['sharedlayers']) > 0:
                dense = models.DenseNet(self.config['sharedlayers'], input_shape=input[-1],
                                        scale=self.config['scaleobs'], final_act=True,
                                        activation=self.config['activation'], batch_norm=self.config["batch_norm"],
                                        init_weight=self.config["init_weight"])
                num_features = self.config['sharedlayers'][-1]
            else:
                dense = models.ScaledIdentity(self.config['scaleobs'])
                num_features = input[-1]
        if state_dict:
            dense.load_state_dict(state_dict)
            logger.info('sharednet loaded')
        return dense, num_features

    def Qnet(self, input, state_dict=None):
        layers = self.config['hiddenlayers']
        Q = models.DenseNet(layers + [self.n_out], input_shape=input, scale=1, final_act=False,
                            activation=self.config['activation'], batch_norm=self.config["batch_norm"],
                            init_weight=self.config["init_weight"])
        if state_dict:
            Q.load_state_dict(state_dict)
            logger.info('Q loaded')
        return Q

    def policyNet(self, input, state_dict=None):
        layers = self.config['hiddenlayers']
        logitpolicy = models.DenseNet(layers + [self.n_out], input_shape=input,
                                      scale=1, activation=self.config['activation'],
                                      batch_norm=self.config["batch_norm"],
                                      init_weight=self.config["init_weight"])
        if state_dict:
            logitpolicy.load_state_dict(state_dict)
            logger.info('policyNet loaded')
        return logitpolicy

    def Vnet(self, input, state_dict=None):
        layers = self.config['hiddenlayers']
        V = models.DenseNet(layers + [1], input_shape=input, final_act=False,
                            scale=1, activation=self.config['activation'], batch_norm=self.config["batch_norm"],
                            init_weight=self.config["init_weight"])
        if state_dict:
            V.load_state_dict(state_dict)
            logger.info('V loaded')
        return V

    def transition_net(self, len_shared_features, actions, state_dict=None):
        layers = [128, len_shared_features]  # fixme
        m = models.DenseNet(layers, input_shape=len_shared_features + actions, final_act=False,
                            scale=1, activation=self.config['activation'], batch_norm=self.config["batch_norm"],
                            init_weight=self.config["init_weight"])
        if state_dict:
            m.load_state_dict(state_dict)
            logger.info('Tr loaded')
        return m

    def initQnetwork(self):
        self.models = {}
        if self.config["path_exp"] is not None and os.path.isfile(self.config["path_exp"] + ".pth"):
            device = None if self.config['use_cuda'] else 'cpu'
            checkpoint = torch.load(self.config["path_exp"] + ".pth", map_location=device)
            logger.info('checkpoint loaded')
        else:
            logger.info('no checkpoint loaded')
            checkpoint = {'shared': None, 'V': None, 'policy': None, "T": None, "Q": None, "optimizer": None}
        self.learnable_parameters = []
        self.commoninit()
        use_cuda = self.config['use_cuda']
        self.device = torch.device("cuda" if use_cuda else "cpu")
        if self.isdiscrete:
            print('scaled', self.scaled_obs)
            n_input = list(self.scaled_obs[:-1]) + [self.scaled_obs[-1] * (self.config[
                                                                               'past'] + 1)]  # + self.observation_space.shape[0]*onehot(0,len(self.observation_space.shape))*(self.config['past'])
            if self.useConv == False:
                n_input = [int(np.prod(self.scaled_obs + (1 * (self.config['past'] + 1),)))]
            else:
                n_input = [n_input[-1]] + list(n_input[:-1])
            self.n_out = self.action_space.n
            print('n_input', n_input, self.observation_space.shape, self.scaled_obs)
        else:
            raise NotImplemented
        print(self.observation_space, 'obs', n_input, 'action', self.n_out)

        self.shared, self.len_shared_features = self.sharednet(n_input, checkpoint["shared"])
        self.shared = self.shared.to(self.device, non_blocking=True)
        self.models["shared"] = self.shared
        if self.config['policy']:
            self.logitpolicy = self.policyNet(self.len_shared_features, checkpoint["policy"])
            self.V = self.Vnet(self.len_shared_features, checkpoint["V"])
            self.logitpolicy = self.logitpolicy.to(self.device, non_blocking=True)
            self.V = self.V.to(self.device, non_blocking=True)

            self.models["V"] = self.V
            self.models["policy"] = self.logitpolicy

            self.learnable_parameters += self.logitpolicy.parameters()
            self.learnable_parameters += self.V.parameters()
        else:
            self.Q = self.Qnet(self.len_shared_features, checkpoint["Q"]).to(self.device, non_blocking=True)
            if 'transition_net' in self.config and self.config['transition_net']:
                self.state_list = [[], []]
                self.avg_loss_trans = None
                self.T = self.transition_net(self.len_shared_features, self.n_out, checkpoint["T"])
                self.T = self.T.to(self.device, non_blocking=True)
                self.models["T"] = self.T
                self.learnable_parameters += self.T.parameters()

            self.models["Q"] = self.Q
            self.learnable_parameters += self.Q.parameters()

            # this used a shared net for copyQ or doubleQ
            if self.config['copyQ'] > 0 or self.config['doubleQ']:
                self.copy_shared, _ = self.sharednet(n_input, checkpoint["shared"])
                self.copy_shared = self.copy_shared.to(self.device, non_blocking=True)
                self.copy_Q = self.Qnet(self.len_shared_features, checkpoint["Q"]).to(self.device, non_blocking=True)

        if self.config['normalize']:
            self.avg_target = None

        self.learnable_parameters += self.shared.parameters()

        if self.config['policy']:
            reduction = 'mean'
            self.policy_criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            reduction = 'none'
        if self.config['loss'].lower() == 'mse':
            self.criterion = nn.MSELoss(reduction=reduction)
        elif self.config['loss'].lower() == 'clipmse':
            self.criterion = nn.SmoothL1Loss(reduction=reduction)
        else:
            raise NotImplemented

        if self.config['policy'] == False:
            if self.config['doubleQ']:
                raise NotImplementedError
                # fixme to remove (old tf implementation)
                self.singleQ2 = tf.reduce_sum(self.curraction * self.Q2,
                                              reduction_indices=1)  # tf.reduce_sum(input_tensor, reduction_indices, keep_dims, name)#tf.slice(self.Q, [0,self.curraction],[-1,1])  #self.Q[:,self.out]
                with tf.variable_scope('copyQ', reuse=True):
                    self.Q2next = self.fullQNet(self.nextstate, reuse=True)
                argmaxQnext = tf.cast(tf.argmax(self.Qnext, 1), dtype=tf.int32)
                argmaxQ2next = tf.cast(tf.argmax(self.Q2next, 1), dtype=tf.int32)
                cat_idx2 = tf.stack([tf.range(0, tf.shape(self.Qnext)[0]), argmaxQ2next], axis=1)
                cat_idx1 = tf.stack([tf.range(0, tf.shape(self.Qnext)[0]), argmaxQnext], axis=1)
                maxQnext1 = tf.gather_nd(self.Qnext, cat_idx2)
                maxQnext2 = tf.gather_nd(self.Q2next, cat_idx1)
                if self.config['episodic']:
                    delta2 = self.singleQ2 - (
                            self.reward + self.config["discount"] * tf.stop_gradient(maxQnext1) * self.notdone)
                    delta = self.singleQ - (
                            self.reward + self.config["discount"] * tf.stop_gradient(maxQnext2) * self.notdone)
                else:
                    delta2 = self.singleQ2 - (
                            self.reward + self.config["discount"] * tf.stop_gradient(maxQnext1))
                    delta = self.singleQ - (
                            self.reward + self.config["discount"] * tf.stop_gradient(maxQnext2))

                tf.summary.histogram('Q2', self.Q2)
                # assert self.singleQ2.get_shape() == self.y.get_shape()

            if self.config['doubleQ']:
                # fixme to remove (old tf implementation)
                if self.config['clip'] > 0 and "cliptype" in self.config and self.config['cliptype'] == 'deltaclip':
                    self.errorlist2 = clipdelta(delta2, self.config['clip'])
                else:
                    self.errorlist2 = 0.5 * delta2 ** 2

            if self.config['doubleQ']:
                # fixme to remove (old tf implementation)
                if regul != []:
                    logger.warning("doubleQ with regularization not properly implemented")
                self.cost2 = tf.reduce_mean(self.errorlist2)
                self.optimizer2 = tf.train.RMSPropOptimizer(self.learnrate, decay=self.config['decayoptimizer'],
                                                            momentum=self.config['momentum'],
                                                            epsilon=self.config['epsoptimizer'])
                if self.config["clip"] > 0 and self.config['cliptype'] != 'deltaclip':
                    self.optimizer2 = self.clip(self.optimizer2, self.cost2, global_step=self.global_step,
                                                lim=self.config['clip'])
                else:
                    self.optimizer2 = self.optimizer2.minimize(self.cost2, global_step=self.global_step)

        if 'optimizer' in self.config:
            if self.config['optimizer'] == "adam":
                self.optimizer = torch.optim.Adam(self.learnable_parameters, lr=self.learnrate,
                                                  weight_decay=self.config['regularization'],
                                                  eps=self.config['eps_optim'])
            elif self.config['optimizer'] == "sgd":
                self.optimizer = torch.optim.SGD(self.learnable_parameters, lr=self.learnrate,
                                                 weight_decay=self.config['regularization'],
                                                 momentum=self.config['momentum'])
            else:
                self.optimizer = torch.optim.RMSprop(self.learnable_parameters, lr=self.learnrate,
                                                     momentum=self.config['momentum'],
                                                     weight_decay=self.config['regularization'],
                                                     alpha=0.95,
                                                     eps=self.config['eps_optim'])
        else:
            self.optimizer = torch.optim.RMSprop(self.learnable_parameters, lr=self.learnrate,
                                                 momentum=self.config['momentum'],
                                                 weight_decay=self.config['regularization'],
                                                 alpha=0.95,
                                                 eps=self.config['eps_optim'])
        if checkpoint["optimizer"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if 'num_updates' not in self.config:
            self.config['num_updates'] = 0

        if self.config["path_exp"] is not None and (os.path.exists(self.config["path_exp"] + "_mem.p")
                                                    or os.path.exists(self.config["path_exp"] + "_mem.p.zip")):
            self.memory = buffers.load_zipped_pickle(self.config["path_exp"] + "_mem.p")
            logger.info('memory loaded')
        else:
            # self.memory = ReplayMemory(self.config['memsize'],use_priority=self.config['priority_memory'])
            self.memory = buffers.ReplayMemory(self.config['memsize'], self.scaled_obs, self.observation_space.dtype,
                                               self.action_space, self.config['past'], self.config['discount'])
        print((self.config['memsize'],) + tuple(n_input))

    def learn(self, force=False):
        self.update_learning_rate()
        for m in self.models:
            self.models[m].train()
        if self.config['copyQ'] > 0 and self.config['num_updates'] % self.config['copyQ'] == 0:
            logger.debug('copying Q')
            self.copy_shared.load_state_dict(self.shared.state_dict())
            self.copy_Q.load_state_dict(self.Q.state_dict())

        update = (np.random.random() < self.config['probupdate'])
        if update or force:
            if self.memory.sizemem() > self.config['randstart']:
                self.config['num_updates'] += 1
                ind = self.memory.sample(self.config['batch_size'])
                allstate, actions, currew, notdonevec, step_vec, total_reward, step2end = self.memory[ind]
                nextstates, _, _, _, _, _, _ = self.memory[ind + 1]
                allactionsparse = np.eye(self.n_out, dtype=np.float32)[actions]
                if self.config['lambda'] > 0:
                    raise NotImplementedError
                # i = 0
                # #  [0 state, 1 action, 2 reward, 3 notdone, 4 step, 5 total_reward]
                # if self.config['lambda'] > 0:
                #     raise NotImplementedError
                #     #fixme old code won't work anymore
                #     for j in ind:
                #         if (np.random.random() < 0.01 or total_reward[j,0].isnan()):
                #             limitd = 500
                #             gamma = 1.
                #             offset = 0
                #             nextstate = 1  # next state relative index
                #
                #             n = j
                #             while nextstate != None and offset < limitd:
                #                 total_reward[i, 0] += gamma * self.memory[n][2]
                #                 gamma = gamma * self.config['discount']
                #                 if n + nextstate * 2 >= self.memory.sizemem() or not (offset + nextstate < limitd) or \
                #                         self.memory[n][3] == 0:
                #                     total_reward[i, 0] += gamma * self.maxq(self.memory[n + nextstate][0][None,...]) * \
                #                                              self.memory[n][3]
                #                     nextstate = None
                #                 else:
                #                     offset += nextstate
                #                     n = j + offset
                #             self.memory[j][5] = total_reward[i, 0]
                #         else:
                #             total_reward[i, 0] = self.memory[j][5]
                #         i += 1

                flag = np.random.random() < 0.5 and self.fulldouble
                self.optimizer.zero_grad()
                if self.config['doubleQ'] and flag:
                    raise NotImplemented
                    if self.newQ:
                        self.sess.run(self.optimizer2, feed_dict={
                            self.x: allstate,
                            self.reward: currew.reshape((-1,)),
                            self.notdone: notdonevec.reshape((-1,)),
                            self.nextstate: nextstates,
                            self.curraction: allactionsparse})
                    else:
                        self.sess.run(self.optimizer2, feed_dict={
                            self.x: allstate,
                            self.y: alltarget.reshape((-1, 1)),
                            self.curraction: allactionsparse})
                else:
                    allactionsparse = torch.from_numpy(allactionsparse).to(self.device, non_blocking=True)
                    allstate = torch.from_numpy(allstate).to(self.device, non_blocking=True).float()
                    nextstates = torch.from_numpy(nextstates).to(self.device, non_blocking=True).float()
                    currew = torch.from_numpy(currew.reshape((-1,))).to(self.device, non_blocking=True)
                    notdonevec = torch.from_numpy(notdonevec.reshape((-1,))).to(self.device, non_blocking=True)
                    if self.config['lambda'] > 0:
                        total_reward = torch.from_numpy(total_reward.reshape((-1,))).to(self.device, non_blocking=True)
                        step2end = torch.from_numpy(step2end.reshape((-1,))).to(self.device, non_blocking=True)
                    shared_features = self.shared(allstate)
                    if self.config['copyQ'] > 0:
                        next_shared_features = self.copy_shared(nextstates)
                        maxQnext = torch.max(self.copy_Q(next_shared_features), dim=1)[0]
                    else:
                        next_shared_features = self.shared(nextstates)
                        maxQnext = torch.max(self.Q(next_shared_features), dim=1)[0]
                    maxQnext = maxQnext.detach()
                    if self.config['lambda'] > 0:
                        # todo finish
                        total_reward += self.memory.discount ** (step2end + 1) * maxQlast
                    if 'transition_net' in self.config and self.config['transition_net']:
                        pred_next_features_reward = shared_features + self.T(
                            torch.cat((shared_features, allactionsparse), 1))

                    currQ = self.Q(shared_features)
                    singleQ = torch.sum(allactionsparse * currQ, dim=1)

                    if self.config['episodic']:
                        target = (currew + self.config["discount"] * maxQnext * notdonevec) * (
                                1. - self.config['lambda'])
                    else:
                        target = (currew + self.config["discount"] * maxQnext) * (1. - self.config['lambda'])
                    if self.config['lambda'] > 0:
                        target += total_reward * self.config['lambda']

                    if self.config['normalize']:
                        scale_target = 1. * torch.abs(target).mean().detach() + 0.001
                        if self.avg_target is None:
                            self.avg_target = scale_target
                        else:
                            self.avg_target = 0.99 * self.avg_target + 0.01 * scale_target
                        loss = self.criterion(singleQ / self.avg_target,
                                              target / self.avg_target)
                        if np.random.random() < 0.001:
                            print("avg target", self.avg_target.data.item(), "loss", loss.mean().data.item())
                    else:
                        loss = self.criterion(singleQ, target)
                    # if np.random.random() < 0.1:
                    #    print("max loss",loss.max().data.item(),"abs diff",torch.max(torch.abs(singleQ-target)).data.item())
                    if self.config['priority_memory']:
                        alpha = 0.7 * 0.5  # 0.5 because the loss is squared td error
                        self.memory.set_priority(ind, np.minimum((loss ** alpha).cpu().detach().numpy(), 10.))
                        beta = 0.7
                        w = 1.0 / Variable(torch.from_numpy(prob_mem[ind]).float()).to(self.device, non_blocking=True)
                        w = w ** beta
                        w /= w.max()
                        # print(w.min(),w.max(),w.max()/w.min())
                        loss = (loss * w).sum()
                    else:
                        loss = loss.mean()
                    # print("avg",self.avg_target)
                    if 'transition_net' in self.config and self.config['transition_net'] and self.config[
                        'transition_weight'] > 0:
                        target_tr = next_shared_features.detach()  # fixme  #torch.cat((next_shared_features,currew.view(-1,1)/scale_target),1).detach()
                        target_mean = 1. * torch.abs(target_tr).mean() + 0.01
                        # target_mean = target_mean.detach()
                        loss_trans = torch.mean(
                            (pred_next_features_reward / (target_mean) - target_tr / (target_mean)) ** 2)

                        if self.avg_loss_trans is None:
                            self.avg_loss_trans = loss_trans.data.item()
                        self.avg_loss_trans = self.avg_loss_trans * 0.99 + loss_trans.data.item() * 0.01

                        if np.random.random() < 0.01:
                            print("loss_trans", self.avg_loss_trans)
                            print("dist", torch.mean((pred_next_features_reward - target_tr) ** 2).sqrt().data.item())
                        loss += self.config['transition_weight'] * loss_trans

                loss.backward()
                if 'norm_clip' in self.config and self.config['norm_clip']:
                    torch.nn.utils.clip_grad_norm_(self.learnable_parameters, 0.5)
                if 'val_clip' in self.config and self.config['val_clip']:
                    torch.nn.utils.clip_grad_value_(self.learnable_parameters, 1)
                self.optimizer.step()

        return self.config['num_updates']

    def plot_state(self, plt, state_list):
        fig = plt.figure(2)
        fig.canvas.set_window_title(str(self.config["path_exp"]) + " " + str(self.config))

        plt.clf()
        plt.plot(state_list[0][0], state_list[0][1], color='black')
        plt.plot(state_list[1][0], state_list[1][1], color='red')
        fig.canvas.flush_events()

    def learnpolicy(self, startind=None, endind=None):
        self.update_learning_rate()
        for m in self.models:
            self.models[m].train()
        if startind is None or endind is None:
            print('to check')
            startind = 0  # max(0, self.sizemem - self.config['memsize'])
            endind = self.memory.sizemem()
        n = endind - startind
        if n >= 2:  # self.sizemem >= self.config['batch_size']:
            # todo to check
            logger.info('len policy steps {}'.format(n))
            ind = np.arange(startind, endind)
            allstate, actions, currew, notdonevec, step_vec, total_reward, step2end = self.memory[ind]

            # nextstates, _, _, _, _, _ = self.memory[ind + 1]
            # allactionsparse = np.eye(self.n_out, dtype=np.float32)[actions]

            # allstate = np.zeros((n,) + self.memory[startind][0].shape, dtype=np.float32)
            # listdiscounts = np.zeros((n, 1), dtype=np.float32)
            #
            # nextstates = np.zeros((n,) + self.memory[startind][0].shape, dtype=np.float32)
            # currew = np.zeros((n, 1), dtype=np.float32)
            # notdonevec = np.zeros((n, 1), dtype=np.float32)
            # allactions = np.zeros((n,), dtype=np.int64)
            # i = 0
            # #  [0 state, 1 action, 2 reward, 3 notdone, 4 t,5 w]
            # Gtv = np.zeros((n, 1), dtype=np.float32)
            # if self.memory[endind - 1][3] == 1:
            #     # last_state = Variable(torch.from_numpy(self.memory[endind - 1][0].reshape(1,-1)).float()).to(self.device)
            #     totalRv = self.evalV(self.memory[endind - 1][0][None, ...], numpy=True)[0]  # self.V(last_state)[0, 0].cpu().item()
            #     Gtv[0] = totalRv
            # else:
            #     totalRv = self.memory[endind - 1][2]
            #     Gtv[0] = self.memory[endind - 1][2]
            # for j in range(endind - 1, startind - 1, -1):
            #     if i > 0:
            #         totalRv *= self.config['discount']
            #         totalRv += self.memory[j][2]
            #         Gtv[i] = totalRv
            #
            #     listdiscounts[i] = self.config['discount'] ** self.memory[j][4]
            #     allstate[i] = self.memory[j][0]
            #     if self.memory[j][3] == 1 and j + 1 < self.memory.sizemem():
            #         nextstates[i] = self.memory[j + 1][0]
            #
            #     currew[i, 0] = self.memory[j][2]
            #     notdonevec[i, 0] = self.memory[j][3]
            #     allactions[i] = self.memory[j][1]  # , self.n_out)
            #     i += 1
            # assert i == n

            # allstate = Variable(torch.from_numpy(allstate).float()).to(self.device)
            Vallstate = self.evalV(allstate, numpy=False)
            Vnext = torch.cat((Vallstate[1:], torch.zeros(1, 1, device=self.device)), 0).detach()

            notdonevec = torch.from_numpy(notdonevec).to(self.device, non_blocking=True)
            currew = torch.from_numpy(currew).to(self.device, non_blocking=True)
            total_reward = torch.from_numpy(total_reward).to(self.device, non_blocking=True)
            actions = torch.from_numpy(actions).to(self.device, non_blocking=True)
            # Vnext = np.append([[0]],Vallstate[:-1],axis=0)
            if (notdonevec[:-1,0]==0).any():
                raise NotImplementedError
            if self.config['episodic']:
                targetV = currew + self.config['discount'] * Vnext * notdonevec
                if notdonevec[-1, 0] == 1:
                    #todo to check
                    discount_vec = self.memory.discount ** torch.arange(len(total_reward) - 1, -1, -1,
                                                                        device=self.device).float()
                    total_reward += (Vallstate[-1, 0] - currew[-1, 0]) * discount_vec.reshape(-1, 1)
                    targetV[-1] = Vallstate[-1]
                    targetV = targetV[:-1]
                    Vallstate = Vallstate[:-1]
                    total_reward = total_reward[:-1]
                    allstate = allstate[:-1]
                    actions = actions[:-1]

                if np.random.random() < 0.1:
                    # print(notdonevec[0, 0])
                    print('currew', (currew).reshape(-1, )[0:5], (currew).reshape(-1, )[-5:])
                    print('Gtv', (total_reward).reshape(-1, )[0:5], (total_reward).reshape(-1, )[-5:])
                    print('targetV', (targetV).reshape(-1, )[0:5], (targetV).reshape(-1, )[-5:])
                    print('Vstate', Vallstate.reshape(-1, )[0:5], Vallstate.reshape(-1, )[-5:])
                    print("notdonevec", (notdonevec).reshape(-1, )[0:5], (notdonevec).reshape(-1, )[-5:])

                targetV = targetV * (1 - self.config['lambda']) + total_reward * self.config['lambda']

                if self.config['discounted_policy_grad'] == False:
                    listdiscounts = 1.
                else:
                    raise NotImplementedError
                targetp = listdiscounts * (targetV - Vallstate)
                targetp = (targetp - targetp.mean()) / (targetp.std() + 0.00001)
                targetp = targetp.detach()
            else:
                raise Exception('non-episodic not implemented')
                exit(-1)

            self.optimizer.zero_grad()
            logit = self.eval_policy(allstate, numpy=False, logit=True)
            pr, logp = torch.nn.functional.softmax(logit, dim=1), torch.nn.functional.log_softmax(logit, dim=1)

            if np.random.random() < 0.001:
                print('prob', pr, "logit", logit)

            entropy = self.config['entropy'] * torch.mean(-torch.sum(pr * logp, 1))
            print(targetp.shape, self.policy_criterion(logit, actions).shape)
            logpolicy = targetp.view(-1, ) * self.policy_criterion(logit, actions).view(-1, )
            errorpolicy = torch.mean(logpolicy) - entropy

            if self.config['normalize']:
                scale_target = (targetV.detach()**2).mean()
                if self.avg_target is None:
                    self.avg_target = scale_target
                else:
                    self.avg_target = 0.995 * self.avg_target + 0.005 * scale_target
                scaling = torch.sqrt(self.avg_target) + 0.001
                v_loss = self.criterion(Vallstate / scaling, targetV.detach() / scaling)
                if np.random.random() < 0.001:
                    print("avg target", self.avg_target.data.item()**0.5, "v loss", v_loss.mean().data.item())
            else:
                self.avg_target=1
                v_loss = self.criterion(Vallstate, targetV.detach())
            #v_loss = v_loss/(torch.abs(v_loss).detach()+0.01)
            # errorpolicy = errorpolicy/(torch.abs(errorpolicy.detach())+0.01)
            logger.info("error policy {} v loss {} scale_V {}".format(errorpolicy.item(),v_loss.item(),scaling.item()))
            loss = errorpolicy + 3*v_loss  # fixme
            if torch.isnan(logpolicy).any():
                print("a", actions, logpolicy, targetp, "logit", logit)
                raise Exception('error logpolicy')
            loss.backward()
            self.optimizer.step()

            return True
        else:
            return False

    def maxq(self, observation):
        for m in self.models:
            self.models[m].eval()
        assert observation.ndim > 1
        assert self.isdiscrete

        if self.copyQalone:
            var_obs = torch.from_numpy(observation).float().to(self.device, non_blocking=True)
            currQ = self.copy_Q(self.copy_shared(var_obs)).cpu().data.numpy()
            return np.max(currQ).reshape(1, )
        else:
            var_obs = torch.from_numpy(observation).float().to(self.device, non_blocking=True)
            currQ = self.Q(self.shared(var_obs)).cpu().data.numpy()
            return np.max(currQ).reshape(1, )

    def maxqbatch(self, observation):
        raise NotImplementedError
        if self.isdiscrete:
            if self.copyQalone:
                return np.max(self.sess.run(self.Q2, feed_dict={self.x: observation}), 1)
            else:
                return np.max(self.sess.run(self.Q, feed_dict={self.x: observation}), 1)

    def doublemaxqbatch(self, observation, flag):
        raise NotImplementedError
        # print observation
        if self.isdiscrete:
            if flag:
                Q = self.sess.run(self.Q, feed_dict={self.x: observation})
                # print Q.shape,np.argmax(self.sess.run(self.Q2, feed_dict={self.x: observation}),1).shape,Q[np.argmax(self.sess.run(self.Q2, feed_dict={self.x: observation}),1)].shape
                return Q[range(Q.shape[0]), np.argmax(self.sess.run(self.Q2, feed_dict={self.x: observation}), 1)]
            else:
                Q2 = self.sess.run(self.Q2, feed_dict={self.x: observation})
                # print Q2.shape,np.argmax(self.sess.run(self.Q, feed_dict={self.x: observation}),1).shape, Q2[np.argmax(self.sess.run(self.Q, feed_dict={self.x: observation}),1)].shape
                return Q2[range(Q2.shape[0]), np.argmax(self.sess.run(self.Q, feed_dict={self.x: observation}), 1)]

    def argmaxq(self, observation):
        for m in self.models:
            self.models[m].eval()
        if self.isdiscrete:
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            else:
                observation = observation.reshape(tuple([1] + list(observation.shape)))
            # print observation,self.sess.run(self.Q, feed_dict={self.x:observation})
            if self.config['doubleQ'] and (self.fulldouble and np.random.random() < 0.5):
                return np.argmax(self.sess.run(self.Q2, feed_dict={self.x: observation}))
            else:
                var_obs = torch.from_numpy(observation).to(self.device, non_blocking=True).float()
                shared_features = self.shared(var_obs)
                if np.random.random() < 0.001:
                    logger.debug("shared_features {}".format(shared_features.cpu().data.numpy().reshape(-1, )[:100]))
                currQ = self.Q(shared_features).cpu().data.numpy()
                best_action = np.argmax(currQ)

                return best_action
                # return np.argmax(self.sess.run(self.Q, feed_dict={self.x: observation}))

    def evalQ(self, observation):
        # todo add with torch.no_grad(): at every eval
        for m in self.models:
            self.models[m].eval()
        assert observation.ndim > 1
        var_obs = Variable(torch.from_numpy(observation).float()).to(self.device, non_blocking=True)
        currQ = self.Q(self.shared(var_obs)).cpu().data.numpy()  # fixme is this necessary?
        return currQ

    def eval_policy(self, observation, numpy, logit=False):
        for m in self.models:
            self.models[m].eval()
        if self.isdiscrete:
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
        input = torch.from_numpy(observation).float()
        input = input.to(self.device, non_blocking=True)
        if logit:
            prob = self.logitpolicy(self.shared(input))
        else:
            lg = self.logitpolicy(self.shared(input))
            if torch.isnan(lg).any():
                print("nan", lg, "\n", input, "\n", self.shared(input))
            assert torch.isnan(lg).any() == False
            prob = torch.nn.functional.softmax(lg)
        if numpy:
            return prob.cpu().data.numpy()
        else:
            return prob

    def evalV(self, observation, numpy):
        for m in self.models:
            self.models[m].eval()
        assert observation.ndim > 1
        input = torch.from_numpy(observation).float()
        input = input.to(self.device, non_blocking=True)
        v = self.V(self.shared(input))
        if numpy:
            return v.cpu().data.numpy()
        else:
            return v

    def softmaxq(self, observation):
        for m in self.models:
            self.models[m].eval()
        if self.isdiscrete:
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            G = self.sess.run(self.Q, feed_dict={self.x: observation})
            p = np.exp(G * 5) / np.sum(np.exp(G * 5))
            p = p.reshape(-1, )
            return np.random.choice(p.shape[0], 1, p=p)[0]
        else:
            print('not implemented')
            exit(0)

    def act(self, observation, episode=None, update_state=False):
        for m in self.models:
            self.models[m].eval()
        eps = self.epsilon(episode)

        # epsilon greedy.
        if np.random.random() > eps:
            action = self.argmaxq(observation)  # self.softmaxq(observation)#
        # print self.softmaxq(observation)
        # print 'greedy',action
        else:
            action = self.action_space.sample()
        # print 'sample',action

        if update_state:  # fixme check
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            else:
                observation = observation.reshape(tuple([1] + list(observation.shape)))
            var_obs = Variable(torch.from_numpy(observation).float()).to(self.device, non_blocking=True)
            shared_features = self.shared(var_obs)
            best_action_onehot = onehot(action, self.n_out).reshape(1, -1)
            best_action_onehot = Variable(torch.from_numpy(best_action_onehot).float()).to(self.device,
                                                                                           non_blocking=True)
            self.state_list[0].append(shared_features.cpu().data.numpy())
            in_tr = torch.cat((shared_features, best_action_onehot), 1)
            # print(in_tr.shape)
            self.state_list[1].append((shared_features + self.T(in_tr)).cpu().data.numpy())

        return action

    def actpolicy(self, observation, episode=None):
        for m in self.models:
            self.models[m].eval()
        prob = self.eval_policy(observation[None, ...], numpy=True)[0]
        if episode is None or episode < 0:
            action = np.argmax(prob)
        else:
            action = np.random.choice(self.n_out, p=prob)  # (prob[0]+0.04)/np.sum(prob[0]+0.04))
        assert np.isnan(prob).any() == False
        if np.random.random() < 0.001:
            print('prob', prob)
        return action

    def update_learning_rate(self):
        self.learnrate = self.config['initial_learnrate'] * self.config['decay_learnrate'] ** (
                    self.config['num_updates'] / 1000000.0)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learnrate
