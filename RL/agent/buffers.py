import gym
import numpy as np
import random
import torch
class ReplayMemory(object):
    def __init__(self, max_size, observation_dims, observation_dtype, action_space: gym.Space, history: int, use_priority=False):
        if len(list(observation_dims))==3:
            assert observation_dims[2]==1 # assuming 1 channel
            observation_dims = list(observation_dims[:2])
            self.reshape = False
        elif history>0:
            self.reshape = True
        else:
            self.reshape = False
        self.obs_mem = np.zeros([max_size] + list(observation_dims), dtype=observation_dtype)
        self.action_mem = np.zeros([max_size] + list(action_space.shape), dtype=action_space.dtype)
        self.info_mem = []
        self.history = history
        assert history>=0
        self.current_size = 0
        self.max_size = max_size
        self.last_ind = -1
        self.start_ind = -1
        self.episodes = []
        self.use_priority = use_priority
        if self.use_priority:
            raise NotImplementedError
            self.max_priority = 0.0000001
            self.priority = np.zeros(max_size) + self.max_priority

    def empty(self):
        self.last_ind = -1
        self.start_ind = -1
        self.current_size = 0
        self.info_mem = []
        self.episodes = []

    def __getitem__(self, item):  # item has to be from 0 to len(mem)-1
        assert (item < self.current_size and item>=self.history) # change to >=0 for policy #TODO
        if self.history>0:
            if item<self.history:
                assert len(self.info_mem)<=self.last_ind+1 # to check fixme (maybe used only in policy learning
                idx_list = [0]*(self.history-item) + list(range(0,item+1))
            else:
                idx_list = list(range(item-self.history,item+1))
            #assert len(idx_list) == self.history + 1
            idx = [(self.start_ind + i + self.max_size) % self.max_size for i in idx_list]
            val = [self.obs_mem[idx], self.action_mem[idx[-1]]] + self.info_mem[idx[-1]]
        else:
            idx = (self.start_ind + item + self.max_size) % self.max_size
            val = [self.obs_mem[idx], self.action_mem[idx]] + self.info_mem[idx]
        if self.reshape:
            val[0] = val[0].reshape(-1)
        #print(val[0].shape)
        return val

    def set_priority(self, idx, vals):  # item has to be from 0 to len(mem)-1
        raise NotImplementedError
        assert (idx < len(self.mem)).all()
        idx_new = (self.start_ind + idx) % self.max_size
        self.priority[idx_new] = vals

    def get_priorities(self):
        raise NotImplementedError
        assert self.start_ind>=0 and self.last_ind>=0
        if self.start_ind<=self.last_ind:
            return self.priority[self.start_ind:self.last_ind] #last element is not returned! (because there is no next state)
        else:
            return np.concatenate((self.priority[self.start_ind:],self.priority[:self.last_ind]))

    def update_max(self):
        raise NotImplementedError
        if self.sizemem()>2:
            pr = self.get_priorities()
            self.max_priority = np.minimum(pr.max(),pr.mean()+4*pr.std())

    def add(self, example):
        if self.use_priority and random.random()<0.05:
            raise NotImplementedError
            self.update_max()
            if random.random()<0.1:
                print("max priority",self.max_priority,self.get_priorities().mean())
        self.last_ind += 1
        self.last_ind = self.last_ind % self.max_size
        self.current_size = min(self.current_size+1,self.max_size)
        self.obs_mem[self.last_ind] = example[0]
        self.action_mem[self.last_ind] = example[1]
        if len(self.info_mem)<=self.last_ind:
            self.info_mem.append(example[2:])
            #assert self.sizemem() == len(self.info_mem)
            self.start_ind = 0
        else:
            #assert self.sizemem()==self.max_size
            self.start_ind = (self.last_ind + 1) % self.max_size
            self.info_mem[self.last_ind] = example[2:]

    def sizemem(self):
        return self.current_size