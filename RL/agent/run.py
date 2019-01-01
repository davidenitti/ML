'''

@author: Davide Nitti
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import common
import default_params

import time
import numpy as np
import gym
import gym.spaces

import torchagent
import agent_utils
import argparse
import os
import json

def loadparams(filename):
    with open(filename + ".json", "r") as input_file:
        out=json.load(input_file)
    return out

def getparams(episodes=1000000):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default="")
    args = parser.parse_args()

    if args.file!="" and os.path.isfile(args.file + ".json"):
        params=loadparams(args.file)
        params["file"] = args.file
    else:
        options={
            'target':"LunarLander-v2",#LunarLander-v2 Breakout-v0
            'episodes':episodes,
            'render':True,
            'plot':True,
            'monitor':False,
            'logging':1
        }
        params = default_params.get_default(options['target'])
        params.update(options)

    return params

def main(numavg=100,params=None, sleep=0.0):
    if params is None:
        params = getparams()

    if params['plot'] != True:
        import matplotlib
        matplotlib.use('pdf')
    else:
        import matplotlib
        #matplotlib.use('Agg')
        #matplotlib.use("Qt5agg")
        import matplotlib.pyplot as plt

        plt.rcParams['image.interpolation'] = 'nearest'

    nameenv = params['target']

    reward_threshold = gym.envs.registry.spec(nameenv).reward_threshold
    env = gym.make(nameenv)

    resultsdir = "./data" + nameenv

    if params['monitor'] == True: # store performance and video (check https://gym.openai.com/docs )
        from gym import wrappers
        env = wrappers.Monitor(env, resultsdir, force=True)


    logger = common.init_log(params)

    logger.info(str((env.observation_space, env.action_space, env.spec.timestep_limit, env.reward_range,
                     gym.envs.registry.spec(nameenv).trials)))
    for p in params:
        logger.debug(p+" "+str(params[p]))

    if params["seed"] > 0:
        env.seed(params["seed"])
        np.random.seed(params["seed"])
        logger.debug("seed "+str(params["seed"]))
    try:
        agent = torchagent.deepQconv(logger, env.observation_space, env.action_space, env.reward_range,
                                 **params)
        num_steps = env.spec.timestep_limit
        avg = None
        plt.ion()

        totrewlist = []
        greedyrewlist = [[],[]]
        totrewavglist = []
        total_rew_discountlist = []
        testevery = 20
        useConv=agent.config['conv']

        print(agent.config)
        start_episode = agent.config['start_episode']
        agent.start_async_plot([(totrewlist,totrewavglist,greedyrewlist),reward_threshold], plt,params['plot'] == True,start_episode=start_episode)

        if agent.config['threads'] > 0:
            agent.start_async_learning(num_threads=None,delay=agent.config['delay_threads'])

        for episode in range(start_episode,params['episodes']):

            if (episode) % testevery == 0 or episode >= params['episodes'] - numavg:
                is_test=True
            else:
                is_test=False

            if is_test:
                render = (params['render'])
                eps = -1
                learn=False
                print(agent.config['file'],'episode', episode, 'l rate', agent.getlearnrate(),'lambda',agent.config['lambda'])
            else:
                render = False
                learn = True
                eps = episode
            startt = time.time()
            total_rew, steps, total_rew_discount = agent_utils.do_rollout(agent, env, eps, num_steps=num_steps,
                            render=render,  useConv=useConv,
                            discount=agent.config["discount"], sleep=sleep, learn=learn)
            if avg is None:
                avg = total_rew
            if is_test:
                greedyrewlist[0].append(total_rew / agent.config['scalereward'])
                greedyrewlist[1].append(episode)
                inc = max(0.2,0.05+ 1. / (episode + 1.) ** 0.5)
                avg = avg * (1 - inc) + inc * total_rew
                totrewavglist.append(avg / agent.config['scalereward'])

            if episode % 10 == 0:
                print(agent.config)
            if (episode+1-start_episode) % 20 == 0:
                if agent.config['file'] is not None:
                    print("saving...")
                    agent.config['start_episode'] = episode
                    agent.save()
            if episode % 1 == 0:

                totrewlist.append(total_rew / agent.config['scalereward'])

                total_rew_discountlist.append(total_rew_discount / agent.config['scalereward'])
            #print(avg,agent.config['scalereward'])
            print("time {:.3f} steps {:6} total reward {:.3f} total reward disc {:.3f} avg {:.3f}, avg100 {:.3f}, eps {:.3f} " \
                    "updates {:8} epoch {:.1f} lr {:.6f}".format((time.time() - startt) / steps * 100.,\
                    steps, total_rew / agent.config['scalereward'],total_rew_discount / agent.config['scalereward'],
                    avg / agent.config['scalereward'],np.mean(np.array(totrewlist[-100:])),\
                    agent.epsilon(eps), agent.config['num_updates'], agent.config['num_updates']/50000, agent.getlearnrate()))
        if params['monitor'] == True:
            env.monitor.close()
        print(agent.config)

        #gym.upload(resultsdir, writeup='https://gist.github.com/davidenitti/c6cd38cd1151ccc0248da8fe1f78e6ee', api_key='sk_7mpGWQjpTmWMv4ym739Xg')# gym.upload(resultsdir, api_key='YOURAPI')
    except Exception as e:
        print('Exception',e)
        agent.coord.request_stop(e)
        agent.stop_threads()
        env.close()
        raise e
    agent.stop_threads()
    env.close()
    return np.mean(totrewlist[-numavg:]),agent.config,totrewlist,totrewavglist,greedyrewlist,reward_threshold

if __name__ == '__main__':
    main()