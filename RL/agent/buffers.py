import gym
import numpy as np
import random
import pickle
import os


class ReplayMemory(object):
    def __init__(self, max_size, observation_dims, observation_dtype,
                 action_space: gym.Space, history: int, use_priority=False):
        assert len(list(observation_dims)) == 3 or len(list(observation_dims))==1
        if len(list(observation_dims)) == 3:
            assert observation_dims[2] == 1  # assuming 1 channel
            observation_dims = list(observation_dims[:2])
            self.reshape = False
        elif history > 0:
            self.reshape = True
        else:
            self.reshape = False
        self.obs_mem = np.zeros([max_size] + list(observation_dims), dtype=observation_dtype)
        self.action_mem = np.zeros([max_size] + list(action_space.shape), dtype=action_space.dtype)
        self.reward_mem = np.zeros([max_size,1], dtype=np.float32)
        self.notdone_mem = np.zeros([max_size,1], dtype=np.float32)
        self.step_mem = np.zeros([max_size,1], dtype=np.int32)
        self.totalr_mem = np.zeros([max_size,1], dtype=np.float32)
        self.totalr_mem[:] = np.nan
        self.info_mem = []
        self.history = history
        assert history >= 0
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
        if isinstance(item, int) or isinstance(item, np.int64):
            item = np.array([item])
        assert (item < self.current_size).all()  # change to >=0 for policy #TODO
        assert (item >= 0).all()
        if self.history > 0:
            idx_list = np.array([np.arange(i - self.history, i + 1) for i in item])
            idx_list = np.maximum(0, idx_list)
            #if min(item) < self.history:
            #    assert len(self.info_mem) <= self.last_ind + 1  # to check fixme (maybe used only in policy learning
            idx = (self.start_ind + idx_list + self.max_size) % self.max_size
            val = [self.obs_mem[idx], self.action_mem[idx[:,-1]], self.reward_mem[idx[:,-1]],
                   self.notdone_mem[idx[:,-1]], self.step_mem[idx[:,-1]], self.totalr_mem[idx[:,-1]]]
        else:
            idx = (self.start_ind + item + self.max_size) % self.max_size
            val = [self.obs_mem[idx], self.action_mem[idx], self.reward_mem[idx],
                   self.notdone_mem[idx], self.step_mem[idx], self.totalr_mem[idx]]
        if self.reshape:
            val[0] = val[0].reshape(val[0].shape[0],-1)
        # print(val[0].shape)
        return val
    # def get_obs_only(self, item):
    #     if isinstance(item, int) or isinstance(item, np.int64):
    #         item = np.array([item])
    #     assert (item < self.current_size).all()  # change to >=0 for policy #TODO
    #     assert (item >= 0).all()
    #     if self.history > 0:
    #         idx_list = np.array([np.arange(i - self.history, i + 1) for i in item])
    #         idx_list = np.maximum(0, idx_list)
    #         if min(item) < self.history:
    #             assert len(self.info_mem) <= self.last_ind + 1  # to check fixme (maybe used only in policy learning
    #         idx = (self.start_ind + idx_list + self.max_size) % self.max_size
    #         val = self.obs_mem[idx]
    #     else:
    #         idx = (self.start_ind + item + self.max_size) % self.max_size
    #         val = self.obs_mem[idx]
    #     if self.reshape:
    #         val = val.reshape(val.shape[0], -1)
    #     return val
    def sample(self, batch_size):
        if self.use_priority:
            prob_mem = self.get_priorities() + 0.000001
            prob_mem /= prob_mem.sum()
            ind = np.random.choice(self.sizemem() - 1, batch_size, p=prob_mem)
        else:
            ind = np.random.choice(self.sizemem() - 1, batch_size)
        return ind

    def set_priority(self, idx, vals):  # item has to be from 0 to len(mem)-1
        raise NotImplementedError
        assert (idx < len(self.mem)).all()
        idx_new = (self.start_ind + idx) % self.max_size
        self.priority[idx_new] = vals

    def get_priorities(self):
        raise NotImplementedError
        assert self.start_ind >= 0 and self.last_ind >= 0
        if self.start_ind <= self.last_ind:
            return self.priority[
                   self.start_ind:self.last_ind]  # last element is not returned! (because there is no next state)
        else:
            return np.concatenate((self.priority[self.start_ind:], self.priority[:self.last_ind]))

    def update_max(self):
        raise NotImplementedError
        if self.sizemem() > 2:
            pr = self.get_priorities()
            self.max_priority = np.minimum(pr.max(), pr.mean() + 4 * pr.std())

    def add(self, obs, action, reward, notdone, step, total_reward, extra_info=[]):
        if self.use_priority and random.random() < 0.05:
            raise NotImplementedError
            self.update_max()
            if random.random() < 0.1:
                print("max priority", self.max_priority, self.get_priorities().mean())
        self.last_ind += 1
        self.last_ind = self.last_ind % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)
        shape = self.obs_mem[self.last_ind].shape
        self.obs_mem[self.last_ind] = obs.reshape(shape)
        self.action_mem[self.last_ind] = action
        self.reward_mem[self.last_ind] = reward
        self.notdone_mem[self.last_ind] = notdone
        self.step_mem[self.last_ind] = step
        self.totalr_mem[self.last_ind] = total_reward
        if len(self.info_mem) <= self.last_ind:
            self.info_mem.append(extra_info)
            # assert self.sizemem() == len(self.info_mem)
            self.start_ind = 0
        else:
            # assert self.sizemem()==self.max_size
            self.start_ind = (self.last_ind + 1) % self.max_size
            self.info_mem[self.last_ind] = extra_info

    def sizemem(self):
        return self.current_size

def save_zipped_pickle(obj, filename, zip=False, protocol=-1):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)
    if zip:
        os.system('zip '+filename+'.zip '+filename)
        os.remove(filename)

def load_zipped_pickle(filename):
    if os.path.exists(filename + '.zip'):
        os.system('unzip '+filename + '.zip')
    with open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
    if os.path.exists(filename + '.zip'):
        os.remove(filename)
    return loaded_object

if __name__ == '__main__':
    mem = ReplayMemory(10, [2,3,1], np.float32, np.array([1]), history=3)
    for i in range(4):
        mem.add(np.array([[i,i+0.5,i],[i,i+0.5,i]]).reshape(2,3,1), np.array([i]), 1,1,i,np.nan)
    save_zipped_pickle(mem,'tmp.mem')
    mem = load_zipped_pickle('tmp.mem')
    idx = mem.sample(2)
    print(idx)
    tmp = mem[idx]
    print(tmp[0])