# modified from baselines/baselines/run.py

import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module

from vel.rl.vecenv.subproc import SubprocVecEnvWrapper
from vel.rl.vecenv.dummy import DummyVecEnvWrapper
from vel.rl.env.classic_atari import ClassicAtariEnv

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}



def build_env(env_id, env_type=None, num_env=1, batch=False, seed=1, reward_scale=1.0,
              gamestate=None, frame_stack=False,logger_dir=None):
    #ncpu = multiprocessing.cpu_count()

    env_type, env_id = get_env_type(env_id, env_type)

    if batch:
        env = make_vec_env(env_id, env_type, num_env, seed, gamestate=gamestate, reward_scale=reward_scale)
        if frame_stack:
            frame_stack_size = 4
            env = VecFrameStack(env, frame_stack_size)
    else:
        assert num_env == 1 or num_env is None
        # assuming stack 4 if frame_stack is true
        env = make_env(env_id, env_type, seed=seed, reward_scale=reward_scale,
                        gamestate=gamestate, wrapper_kwargs={'frame_stack': frame_stack}, logger_dir=logger_dir)

    if env_type == 'mujoco':
        env = VecNormalize(env, use_tf=False)
    return env

def get_env_type(env_id, env_type):
    if env_type is not None:
        return env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    env = build_env('BreakoutNoFrameskip-v4', env_type=None, num_env=1, batch=False, seed=1, reward_scale=1.0, gamestate=None)
    env2 = gym.make('Breakout-v4')
    env2.seed(1)
    print(env.observation_space, env.action_space)
    print(env)
    ob = env.reset()
    ob2 = env2.reset()

    #assert (ob==ob2).all()
    print('ob',ob.shape, 'ob2',ob2.shape)
    for i in range(10):
        a = 0#env.action_space.sample()
        for _ in range(4):
            (obnew, rr, done, _info) = env.step(a)
        (obnew2, rr2, done2, _info2) = env2.step(a)
        obnew2 = cv2.cvtColor(cv2.resize(obnew2, (84, 84), interpolation=cv2.INTER_LINEAR),
                     cv2.COLOR_BGR2GRAY)
        print(obnew[:,:,0].shape, rr, done, _info)
        print(obnew2.shape, rr2, done2, _info2)
        print( abs(obnew[:,:,0]-obnew2).mean())
        print(obnew.repeat(3,2).shape)
        plt.imshow(np.concatenate((obnew.repeat(3,2),obnew2[...,None].repeat(3,2))))
        plt.show()

    exit()
    env = SubprocVecEnvWrapper(
        ClassicAtariEnv('BreakoutNoFrameskip-v4'), frame_history=4
    ).instantiate(parallel_envs=8, seed=1)
    print(env.observation_space, env.action_space, 'max_episode_steps', env.venv.specs[0])
    print(env)
    ob = env.reset()
    print('ob', ob.shape)
    a = np.array([env.action_space.sample() for _ in range(8)])
    print(a)
    (obnew, rr, done, _info) = env.step(a)
    print(obnew.shape, rr.shape, done.shape, _info)
