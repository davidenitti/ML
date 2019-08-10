from numba import jit,jitclass
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
from .agent_utils import ReplayMemory, onehot, vis
import json
import pickle

import copy

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
        # with open(filename + "_mem.p", "wb") as input_file:
        #     pickle.dump(self.memory, input_file)
        checkpoint = {"optimizer": self.optimizer.state_dict()}
        for m in self.model_dict:
            checkpoint[m] = self.model_dict[m].state_dict()
        torch.save(checkpoint, filename + ".pth")

    def __init__(self, observation_space, action_space, reward_range, userconfig):

        if userconfig["path_exp"]:
            self.log_dir = userconfig["path_exp"]
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir)
            os.makedirs(self.log_dir)
        else:
            self.log_dir = None

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
        totrewlist, totrewavglist, greedyrewlist = lists
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
            plt.plot(greedyrewlist[1], totrewavglist, color='blue')
            plt.scatter(greedyrewlist[1], greedyrewlist[0], color='black')
            plt.plot([start_episode + max(0, len(totrewlist) - 1 - 100), start_episode + len(totrewlist) - 1],
                     [np.mean(np.array(totrewlist[-100:])), np.mean(np.array(totrewlist[-100:]))], color='black')
            if self.config["path_exp"]:
                plt.savefig(self.config["path_exp"] + '_reward' + '.png', dpi=400)

            ymin, ymax = plt.gca().get_ylim()
            xmin, xmax = plt.gca().get_xlim()
            plt.text(xmin,ymax, str(self.config), ha='left', wrap=True, fontsize=18)
            if self.config['conv'] and False:  # fixme
                if plot:
                    fig = plt.figure(2)
                    fig.canvas.set_window_title(str(self.config["path_exp"]) + " " + str(self.config))

                self.memory.memoryLock.acquire()
                if self.memory.sizemem() > 1:
                    selec = np.random.choice(self.memory.sizemem() - 1)
                    imageselected = self.memory[selec][0].copy()
                    self.memory.memoryLock.release()
                    filtered1 = np.zeros((1, 1, 10, 10))
                    filtered2 = np.zeros((1, 1, 10, 10))
                    vis(plt, w, imageselected, filtered1[0], filtered2[0])

                    if self.config["path_exp"] and plot == False:
                        plt.savefig(self.config["path_exp"] + '_features' + '.png', dpi=300)

                else:
                    self.memory.memoryLock.release()
            else:
                pass  # fixme
                # self.memoryLock.acquire()
                # if self.memory.sizemem():
                #     selec = np.random.choice(self.memory.sizemem()-1)
                #     stateelected = self.memory[selec][0]#.copy()
                #     # todo add summary
                # self.memoryLock.release()

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
        time.sleep(2.0)

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
            return max(self.config['mineps'], self.config['eps'] * self.config['decay'] ** episode)

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

    def sharednet(self, input):
        convout = None
        if self.useConv:
            dense_conv = models.ConvNet(self.config['convlayers'], input,
                                        activation=self.config['activation'],
                                        # numchannels=1 + 1 * self.config['past'],
                                        scale=self.config['scaleobs'])
            dense_conv_shape = dense_conv(torch.zeros([1] + input)).shape
            if len(self.config['sharedlayers']) > 0:
                dense = models.DenseNet(self.config['sharedlayers'], input_shape=dense_conv_shape[-1],
                                        scale=1, final_act=True,
                                        activation=self.config['activation'],
                                        batch_norm=self.config["batch_norm"])
                dense = nn.Sequential(dense_conv, dense)
                num_features = self.config['sharedlayers'][-1]
            else:
                dense = dense_conv
                num_features = dense_conv_shape[-1]
        else:
            if len(self.config['sharedlayers']) > 0:
                dense = models.DenseNet(self.config['sharedlayers'], input_shape=input[-1],
                                        scale=self.config['scaleobs'], final_act=True,
                                        activation=self.config['activation'], batch_norm=self.config["batch_norm"])
                num_features = self.config['sharedlayers'][-1]
            else:
                dense = models.ScaledIdentity(self.config['scaleobs'])
                num_features = input[-1]
        return convout, dense, num_features

    def Qnet(self, input):
        layers = self.config['hiddenlayers']
        Q = models.DenseNet(layers + [self.n_out], input_shape=input, scale=1, final_act=False,
                            activation=self.config['activation'], batch_norm=self.config["batch_norm"])
        return Q

    def fullQNet(self, input, reuse):  # to remove
        with tf.variable_scope('shared', reuse=reuse):
            _, dense2 = self.sharednet(input)
        with tf.variable_scope('Q', reuse=reuse):
            QQ = self.Qnet(dense2)
        return QQ

    def policyNet(self, input):
        layers = self.config['hiddenlayers']
        logitpolicy = models.DenseNet(layers + [self.n_out], input_shape=input,
                                      scale=1, activation=self.config['activation'],
                                      batch_norm=self.config["batch_norm"])
        return logitpolicy

    def Vnet(self, input):
        layers = self.config['hiddenlayers']
        V = models.DenseNet(layers + [1], input_shape=input, final_act=False,
                            scale=1, activation=self.config['activation'], batch_norm=self.config["batch_norm"])
        return V

    def transition_net(self, len_shared_features, actions):
        layers = [128, len_shared_features]  # fixme
        m = models.DenseNet(layers, input_shape=len_shared_features + actions, final_act=False,
                            scale=1, activation=self.config['activation'], batch_norm=self.config["batch_norm"])
        return m

    def initQnetwork(self):
        self.model_dict = {}
        if self.config["path_exp"] is not None and os.path.isfile(self.config["path_exp"] + ".pth"):
            checkpoint = torch.load(self.config["path_exp"] + ".pth")
        else:
            checkpoint = None
        learnable_parameters = []
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

        self.convout, self.shared, self.len_shared_features = self.sharednet(n_input)

        if self.config['policy']:
            self.logitpolicy = self.policyNet(self.len_shared_features)
            self.V = self.Vnet(self.len_shared_features)

            if checkpoint is not None:
                self.logitpolicy.load_state_dict(checkpoint["policy"])
                self.V.load_state_dict(checkpoint["V"])

            self.logitpolicy = self.logitpolicy.to(self.device,non_blocking=True)
            self.V = self.V.to(self.device,non_blocking=True)

            self.model_dict["V"] = self.V
            self.model_dict["policy"] = self.logitpolicy

            learnable_parameters += self.logitpolicy.parameters()
            learnable_parameters += self.V.parameters()


        else:
            self.Q = self.Qnet(self.len_shared_features)

            if 'transition_net' in self.config and self.config['transition_net']:
                self.state_list = [[], []]
                self.avg_loss_trans = None
                self.T = self.transition_net(self.len_shared_features, self.n_out)
                if checkpoint is not None:
                    self.T.load_state_dict(checkpoint["T"])
                self.T = self.T.to(self.device,non_blocking=True)
                self.model_dict["T"] = self.T
                learnable_parameters += self.T.parameters()

            if checkpoint is not None:
                self.Q.load_state_dict(checkpoint["Q"])

            self.Q = self.Q.to(self.device,non_blocking=True)

            self.model_dict["Q"] = self.Q
            learnable_parameters += self.Q.parameters()

            # this used a shared net for copyQ or doubleQ
            if self.config['copyQ'] > 0 or self.config['doubleQ']:
                self.copy_shared = copy.deepcopy(self.shared).to(self.device,non_blocking=True)
                self.copy_Q = copy.deepcopy(self.Q).to(self.device,non_blocking=True)

        if self.config['normalize']:
            self.avg_target = None
        # if len(self.config['sharedlayers']) > 0:
        if checkpoint is not None:
            self.shared.load_state_dict(checkpoint["shared"])
        self.shared = self.shared.to(self.device,non_blocking=True)
        learnable_parameters += self.shared.parameters()
        self.model_dict["shared"] = self.shared

        if self.config['policy']:
            reduction='mean'
            self.policy_criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            reduction='none'
        if self.config['loss'].lower() == 'mse':
            self.criterion = nn.MSELoss(reduction=reduction)
        elif self.config['loss'].lower() == 'clipmse':
            self.criterion = nn.SmoothL1Loss(reduction=reduction)
        else:
            raise NotImplemented


        if self.config['policy'] == False:
            if self.config['doubleQ']:
                raise NotImplementedError
                #fixme to remove (old tf implementation)
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
            # if self.config['clip'] > 0 and "cliptype" in self.config and self.config['cliptype'] == 'deltaclip':
            #     self.errorlist = clipdelta(delta, self.config['clip'])
            #     print(self.errorlist.get_shape())
            # else:
            #     pass  # self.errorlist = 0.5 * delta ** 2  # tf.minimum(0.5 * delta ** 2, tf.abs(delta))  # (self.singleQ - self.y) ** 2
            # print('errorlist', self.errorlist.get_shape())
            if self.config['doubleQ']:
                # fixme to remove (old tf implementation)
                if self.config['clip'] > 0 and "cliptype" in self.config and self.config['cliptype'] == 'deltaclip':
                    self.errorlist2 = clipdelta(delta2, self.config['clip'])
                else:
                    self.errorlist2 = 0.5 * delta2 ** 2
            # self.cost = tf.reduce_mean(self.errorlist)

            # regul=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,tf.get_variable_scope().name)
            # printRegularization()
            # if regul!=[]:
            #    self.cost+= tf.add_n(regul)
            # tf.summary.histogram('cost', self.cost)
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
                self.optimizer = torch.optim.Adam(learnable_parameters, lr=self.learnrate,
                                              weight_decay=self.config['regularization'],
                                              eps=self.config['eps_optim'])
            elif self.config['optimizer'] == "sgd":
                self.optimizer = torch.optim.SGD(learnable_parameters, lr=self.learnrate,
                                                 weight_decay=self.config['regularization'],
                                                 momentum=self.config['momentum'])
            else:
                raise NotImplementedError
        else:
            self.optimizer = torch.optim.RMSprop(learnable_parameters, lr=self.learnrate,
                                                 momentum=self.config['momentum'],
                                                 weight_decay=self.config['regularization'],
                                                 alpha=0.95,
                                                 eps=self.config['eps_optim'])
        if checkpoint is not None:
           self.optimizer.load_state_dict(checkpoint["optimizer"])

        if 'num_updates' not in self.config:
            self.config['num_updates'] = 0

        self.memory = ReplayMemory(self.config['memsize'],use_priority=self.config['priority_memory'])

        print((self.config['memsize'],) + tuple(n_input))

    def learn(self, force=False, print_iteration=False):
        for m in self.model_dict:
            self.model_dict[m].train()
        if self.config['copyQ'] > 0 and self.config['num_updates'] % self.config['copyQ'] == 0:
            #print('copying Q')
            self.copy_shared.load_state_dict(self.shared.state_dict())
            self.copy_Q.load_state_dict(self.Q.state_dict())

        update = (np.random.random() < self.config['probupdate'])
        if update or force:
            self.memory.memoryLock.acquire()
            if self.memory.sizemem() > self.config['randstart']:
                self.config['num_updates'] += 1
                if self.config['priority_memory']:
                    prob_mem = self.memory.get_priorities()+0.000001
                    prob_mem /= prob_mem.sum()
                    ind = np.random.choice(self.memory.sizemem()-1, self.config['batch_size'],p=prob_mem)
                else:
                    ind = np.random.choice(self.memory.sizemem()-1, self.config['batch_size'])
                allstate = np.zeros((ind.shape[0],) + self.memory[ind[0]][0].shape)
                nextstates = np.zeros((ind.shape[0],) + self.memory[ind[0]][0].shape)
                currew = np.zeros((ind.shape[0], 1))
                maxsteps_reward = np.zeros((ind.shape[0], 1))
                notdonevec = np.zeros((ind.shape[0], 1))
                allactionsparse = np.zeros((ind.shape[0], self.n_out))
                i = 0
                #  [0 state, 1 action, 2 reward, 3 notdone, 4 step, 5 total_reward]

                for j in ind:
                    allstate[i] = self.memory[j][0]
                    if self.memory[j][3] == 1:
                        nextstates[i] = self.memory[j + 1][0]
                        # assert (nextstates[i]==self.memory[j][-1]).all()
                    currew[i, 0] = self.memory[j][2]
                    notdonevec[i, 0] = self.memory[j][3]
                    allactionsparse[i] = onehot(self.memory[j][1], self.n_out)

                    if self.config['lambda'] > 0 and (np.random.random() < 0.01 or self.memory[j][5] is None):
                        limitd = 500
                        gamma = 1.
                        offset = 0
                        nextstate = 1  # next state relative index

                        n = j
                        while nextstate != None and offset < limitd:
                            maxsteps_reward[i, 0] += gamma * self.memory[n][2]
                            gamma = gamma * self.config['discount']
                            if n + nextstate * 2 >= self.memory.sizemem() or not (offset + nextstate < limitd) or \
                                            self.memory[n][3] == 0:
                                maxsteps_reward[i, 0] += gamma * self.maxq(self.memory[n + nextstate][0]) * \
                                                         self.memory[n][3]
                                nextstate = None
                            else:
                                offset += nextstate
                                n = j + offset
                        self.memory[j][5] = maxsteps_reward[i, 0]
                    else:
                        maxsteps_reward[i, 0] = self.memory[j][5]
                    i += 1

                self.memory.memoryLock.release()
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
                    allactionsparse = torch.from_numpy(allactionsparse).float().to(self.device,non_blocking=True)
                    allstate = torch.from_numpy(allstate).float().to(self.device,non_blocking=True)
                    nextstates = torch.from_numpy(nextstates).float().to(self.device,non_blocking=True)
                    currew = torch.from_numpy(currew.reshape((-1,))).float().to(self.device,non_blocking=True)
                    notdonevec = torch.from_numpy(notdonevec.reshape((-1,))).float().to(self.device,non_blocking=True)
                    maxsteps_reward = torch.from_numpy(maxsteps_reward.reshape((-1,))).float().to(self.device,non_blocking=True)
                    shared_features = self.shared(allstate)
                    if self.config['copyQ'] > 0:
                        next_shared_features = self.copy_shared(nextstates)
                        maxQnext = torch.max(self.copy_Q(next_shared_features), dim=1)[0]
                    else:
                        next_shared_features = self.shared(nextstates)
                        maxQnext = torch.max(self.Q(next_shared_features), dim=1)[0]
                    maxQnext = maxQnext.detach()

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
                        target += maxsteps_reward * self.config['lambda']

                    if self.config['normalize']:
                        scale_target = 1. * torch.abs(target).mean().detach() + 0.001
                        if self.avg_target is None:
                            self.avg_target = scale_target
                        else:
                            self.avg_target = 0.99 * self.avg_target + 0.01 * scale_target
                        loss = self.criterion(singleQ / self.avg_target,
                                              target / self.avg_target)
                        if np.random.random() < 0.001:
                            print("avg target", self.avg_target.data.item(),"loss",loss.mean().data.item())
                    else:
                        loss = self.criterion(singleQ, target)
                    #if np.random.random() < 0.1:
                    #    print("max loss",loss.max().data.item(),"abs diff",torch.max(torch.abs(singleQ-target)).data.item())
                    if self.config['priority_memory']:
                        alpha = 0.7*0.5 # 0.5 because the loss is squared td error
                        self.memory.set_priority(ind, np.minimum((loss**alpha).cpu().detach().numpy(),10.))
                        beta = 0.7
                        w = 1.0/Variable(torch.from_numpy(prob_mem[ind]).float()).to(self.device,non_blocking=True)
                        w = w**beta
                        w /= w.max()
                        #print(w.min(),w.max(),w.max()/w.min())
                        loss = (loss*w).sum()
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
                self.optimizer.step()

                # if checkcost:
                #     costafter = self.sess.run(self.cost,feed_dict={self.x: allstate, self.y: alltarget.reshape((-1, 1)),
                #                                          self.curraction: allactionsparse})
                #     print('cost after-before',costafter-costbefore,alltarget[0:5], max(0,self.config['memsize']-self.sizemem))
                #
                #     if costafter-costbefore>0 and self.learnrate>self.config["initial_learnrate"]/200:
                #         self.learnrate *=0.97
                #     elif self.learnrate<self.config["initial_learnrate"]*5:
                #         self.learnrate *=1.01
                # #if np.random.random() <0.01:
                # #    print('learning',self.sess.run(self.Q,alltarget[0:5].reshape(-1, ))
            else:
                self.memory.memoryLock.release()
        return self.config['num_updates']

    def plot_state(self, state_list):  # fixme use plt instead of plt0
        fig = plt0.figure(2)
        fig.canvas.set_window_title(str(self.config["path_exp"]) + " " + str(self.config))

        plt0.clf()
        plt0.plot(state_list[0][0], state_list[0][1], color='black')
        plt0.plot(state_list[1][0], state_list[1][1], color='red')
        # plt0.draw()
        # plt0.pause(0.0001)
        fig.canvas.flush_events()

    def learnpolicy(self, startind=None, endind=None):
        for m in self.model_dict:
            self.model_dict[m].train()
        self.memory.memoryLock.acquire()
        # todo fix indexing with new indexes
        if startind is None or endind is None:
            print('to check')
            startind = 0  # max(0, self.sizemem - self.config['memsize'])
            endind = self.memory.sizemem()
        n = endind - startind
        if n >= 1:  # self.sizemem >= self.config['batch_size']:
            allstate = np.zeros((n,) + self.memory[startind][0].shape)
            listdiscounts = np.zeros((n, 1))

            nextstates = np.zeros((n,) + self.memory[startind][0].shape, dtype=np.float32)
            currew = np.zeros((n, 1), dtype=np.float32)
            notdonevec = np.zeros((n, 1), dtype=np.float32)
            allactions = np.zeros((n,), dtype=np.int64)
            i = 0
            #  [0 state, 1 action, 2 reward, 3 notdone, 4 t,5 w]
            Gtv = np.zeros((n, 1), dtype=np.float32)
            if self.memory[endind - 1][3] == 1:
                # last_state = Variable(torch.from_numpy(self.memory[endind - 1][0].reshape(1,-1)).float()).to(self.device)
                totalRv = self.evalV(self.memory[endind - 1][0][None,...], numpy=True)[
                    0]  # self.V(last_state)[0, 0].cpu().item()

                Gtv[0] = totalRv
            else:
                totalRv = self.memory[endind - 1][2]
                Gtv[0] = self.memory[endind - 1][2]
            for j in range(endind - 1, startind - 1, -1):
                if i > 0:
                    totalRv *= self.config['discount']
                    totalRv += self.memory[j][2]
                    Gtv[i] = totalRv

                listdiscounts[i] = self.config['discount'] ** self.memory[j][4]
                allstate[i] = self.memory[j][0]
                if self.memory[j][3] == 1 and j + 1 < self.memory.sizemem():
                    nextstates[i] = self.memory[j + 1][0]

                currew[i, 0] = self.memory[j][2]
                notdonevec[i, 0] = self.memory[j][3]
                allactions[i] = self.memory[j][1]  # , self.n_out)
                i += 1
            assert i == n

            # allstate = Variable(torch.from_numpy(allstate).float()).to(self.device)
            Vallstate = self.evalV(allstate, numpy=False)
            Vnext = torch.cat((Variable(torch.zeros(1, 1, device=self.device)), Vallstate[:-1]), 0).detach()

            notdonevec = Variable(torch.from_numpy(notdonevec)).to(self.device,non_blocking=True)
            currew = Variable(torch.from_numpy(currew)).to(self.device,non_blocking=True)
            Gtv = Variable(torch.from_numpy(Gtv)).to(self.device,non_blocking=True)
            allactions = Variable(torch.from_numpy(allactions)).to(self.device,non_blocking=True)
            # Vnext = np.append([[0]],Vallstate[:-1],axis=0)
            if self.config['episodic']:
                targetV = currew + self.config['discount'] * Vnext * notdonevec
                # targetQ = currew + self.config['discount'] * self.maxqbatch(nextstates).reshape(-1, 1) * notdonevec

                if notdonevec[0, 0] == 1:
                    targetV[0] = Vallstate[0]
                    targetV = targetV[1:]
                    Vallstate = Vallstate[1:]
                    listdiscounts = listdiscounts[1:]
                    Gtv = Gtv[1:]
                    allstate = allstate[1:]
                    allactions = allactions[1:]

                if np.random.random() < 0.2:
                    print(notdonevec[0, 0])
                    print('currew', (currew).reshape(-1, )[0:5], (currew).reshape(-1, )[-5:])
                    print('Gtv', (Gtv).reshape(-1, )[0:5], (Gtv).reshape(-1, )[-5:])
                    print('targetV', (targetV).reshape(-1, )[0:5], (targetV).reshape(-1, )[-5:])
                    print('Vstate', Vallstate.reshape(-1, )[0:5],Vallstate.reshape(-1, )[-5:])
                    print((notdonevec).reshape(-1, )[0:5], (notdonevec).reshape(-1, )[-5:])

                targetV = targetV * (1 - self.config['lambda']) + Gtv * self.config['lambda']
                # print(Gtv,targetV)
                # targetQ = targetQ*(1-self.config['lambda']) + Gt*self.config['lambda']
                if self.config['discounted_policy_grad'] == False:
                    listdiscounts = 1.
                else:
                    listdiscounts = Variable(torch.from_numpy(listdiscounts).float()).to(self.device,non_blocking=True)
                targetp = 1 * listdiscounts * (targetV - Vallstate)
                targetp = (targetp - targetp.mean()) / (targetp.std()+0.000001)
                targetp = targetp.detach()
                # print(targetp.shape)
                # targetp = listdiscounts*(Gt- self.sess.run(self.V, feed_dict={self.x: allstate}))
            else:
                raise Exception('non-episodic not implemented')
                exit(-1)
            self.memory.memoryLock.release()

            self.optimizer.zero_grad()
            logit = self.eval_policy(allstate, numpy=False, logit=True)
            pr, logp = torch.nn.functional.softmax(logit), torch.nn.functional.log_softmax(logit)
            entropy = self.config['entropy'] * torch.mean(-torch.sum(pr * logp, 1))
            logpolicy = targetp.view(-1, ) * self.policy_criterion(logit, allactions).view(-1, )
            errorpolicy = torch.mean(logpolicy) - entropy

            if self.config['normalize']:
                scale_target = 1. * torch.abs(targetV).mean().detach() + 0.001
                if self.avg_target is None:
                    self.avg_target = scale_target
                else:
                    self.avg_target = 0.99 * self.avg_target + 0.01 * scale_target
                v_loss = self.criterion(Vallstate / self.avg_target, targetV.detach() / self.avg_target)
                if np.random.random() < 0.001:
                    print("avg target", self.avg_target.data.item(), "v loss", v_loss.mean().data.item())
            else:
                v_loss = self.criterion(Vallstate, targetV.detach())
            # v_loss = v_loss/v_loss.detach()
            # errorpolicy = errorpolicy/errorpolicy.detach()
            loss = errorpolicy +  3*v_loss  # fixme
            if torch.isnan(logpolicy).any():
                print("a",allactions, logpolicy, targetp,"logit",logit)
            loss.backward()
            self.optimizer.step()

            # self.sess.run(self.optimizer_policy, feed_dict={
            #     self.x: allstate,
            #     self.yp: targetp.reshape((-1, )),
            #     self.y: targetV.reshape((-1, 1)),
            #     self.curraction: allactionsparse})
            #
            return True
        else:
            self.memory.memoryLock.release()
            return False

    def maxq(self, observation):
        for m in self.model_dict:
            self.model_dict[m].eval()
        if self.isdiscrete:
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            else:
                observation = observation.reshape(tuple([1] + list(observation.shape)))
            if self.copyQalone:
                var_obs = Variable(torch.from_numpy(observation).float()).to(self.device,non_blocking=True)
                currQ = self.copy_Q(self.copy_shared(var_obs)).cpu().data.numpy()
                return np.max(currQ).reshape(1, )
            else:
                var_obs = Variable(torch.from_numpy(observation).float()).to(self.device,non_blocking=True)
                currQ = self.Q(self.shared(var_obs)).cpu().data.numpy()
                return np.max(currQ).reshape(1, )

    def maxqbatch(self, observation):
        if self.isdiscrete:
            if self.copyQalone:
                return np.max(self.sess.run(self.Q2, feed_dict={self.x: observation}), 1)
            else:
                return np.max(self.sess.run(self.Q, feed_dict={self.x: observation}), 1)

    def doublemaxqbatch(self, observation, flag):
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
        for m in self.model_dict:
            self.model_dict[m].eval()
        if self.isdiscrete:
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            else:
                observation = observation.reshape(tuple([1] + list(observation.shape)))
            # print observation,self.sess.run(self.Q, feed_dict={self.x:observation})
            if self.config['doubleQ'] and (self.fulldouble and np.random.random() < 0.5):
                return np.argmax(self.sess.run(self.Q2, feed_dict={self.x: observation}))
            else:
                var_obs = Variable(torch.from_numpy(observation).float()).to(self.device,non_blocking=True)
                shared_features = self.shared(var_obs)
                if np.random.random() < 0.001:
                    logger.debug("shared_features {}".format( shared_features.cpu().data.numpy().reshape(-1,)[:100]))
                currQ = self.Q(shared_features).cpu().data.numpy()
                best_action = np.argmax(currQ)

                return best_action
                # return np.argmax(self.sess.run(self.Q, feed_dict={self.x: observation}))

    def evalQ(self, observation):
        for m in self.model_dict:
            self.model_dict[m].eval()
        assert observation.ndim > 1
        var_obs = Variable(torch.from_numpy(observation).float()).to(self.device,non_blocking=True)
        currQ = self.Q(self.shared(var_obs)).cpu().data.numpy()
        return currQ

    def eval_policy(self, observation, numpy, logit=False):
        for m in self.model_dict:
            self.model_dict[m].eval()
        if self.isdiscrete:
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
        input = Variable(torch.from_numpy(observation)).float()
        input = input.to(self.device,non_blocking=True)
        if logit:
            prob = self.logitpolicy(self.shared(input))
        else:
            prob = torch.nn.functional.softmax(self.logitpolicy(self.shared(input)))
        if numpy:
            return prob.cpu().data.numpy()
        else:
            return prob

    def evalV(self, observation, numpy):
        for m in self.model_dict:
            self.model_dict[m].eval()
        assert observation.ndim > 1
        input = Variable(torch.from_numpy(observation)).float()
        input = input.to(self.device,non_blocking=True)
        v = self.V(self.shared(input))
        if numpy:
            return v.cpu().data.numpy()
        else:
            return v

    def softmaxq(self, observation):
        for m in self.model_dict:
            self.model_dict[m].eval()
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
        for m in self.model_dict:
            self.model_dict[m].eval()
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
            var_obs = Variable(torch.from_numpy(observation).float()).to(self.device,non_blocking=True)
            shared_features = self.shared(var_obs)
            best_action_onehot = onehot(action, self.n_out).reshape(1, -1)
            best_action_onehot = Variable(torch.from_numpy(best_action_onehot).float()).to(self.device,non_blocking=True)
            self.state_list[0].append(shared_features.cpu().data.numpy())
            in_tr = torch.cat((shared_features, best_action_onehot), 1)
            # print(in_tr.shape)
            self.state_list[1].append((shared_features + self.T(in_tr)).cpu().data.numpy())

        return action

    def actpolicy(self, observation, episode=None):
        for m in self.model_dict:
            self.model_dict[m].eval()
        prob = self.eval_policy(observation, numpy=True)[0]
        if episode is None or episode < 0:
            action = np.argmax(prob)
        else:
            action = np.random.choice(self.n_out, p=prob)  # (prob[0]+0.04)/np.sum(prob[0]+0.04))
        if np.random.random() < 0.001:
            print('prob', prob)
        return action
