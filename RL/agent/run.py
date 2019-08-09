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
            'target':"CartPole-v0",#LunarLander-v2 Breakout-v0
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
    #print(env.unwrapped.get_action_meanings())
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
                                 params)
        num_steps = env.spec.timestep_limit
        avg = None
        plt.ion()

        totrewlist = []
        greedyrewlist = [[],[]]
        totrewavglist = []
        total_rew_discountlist = []
        testevery = 25
        useConv=agent.config['conv']
        max_total_rew_discount = float("-inf")
        max_abs_rew_discount = float("-inf")

        total_steps = 0
        start_updates = agent.config['num_updates']
        print(agent.config)
        start_episode = agent.config['start_episode']
        agent.start_async_plot([(totrewlist,totrewavglist,greedyrewlist),reward_threshold], plt,params['plot'] == True,start_episode=start_episode)

        if agent.config['threads'] > 0:
            agent.start_async_learning(delay=0.000001)

        for episode in range(start_episode,params['episodes']):

            if (episode+1) % testevery == 0 or episode >= params['episodes'] - numavg:
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
            total_rew, steps, total_rew_discount,max_qval = agent_utils.do_rollout(agent, env, eps, num_steps=num_steps,
                            render=render,  useConv=useConv,
                            discount=agent.config["discount"], sleep=sleep, learn=learn)
            max_total_rew_discount = max(max_total_rew_discount,total_rew_discount)
            max_abs_rew_discount = max(max_abs_rew_discount, abs(total_rew_discount))
            total_steps += steps
            if agent.config['threads'] > 0:
                if ((agent.config['num_updates']-start_updates)/total_steps)<agent.config['probupdate']:
                    sleep+=0.0001
                else:
                    sleep=max(0.,sleep-0.0001)
                print("sleep",sleep,"up",(agent.config['num_updates']-start_updates),"steps",total_steps,
                      'ratio',((agent.config['num_updates']-start_updates)/total_steps))
            if ((max_qval - max_total_rew_discount) / max_abs_rew_discount > 0.9):
                agent.logger.warning("Q function too high: max rew disc  {:.3f}"
                                     " max Q {:.3f} rel error {:.3f}".format(
                                    max_total_rew_discount, max_qval,
                                    (max_qval - max_total_rew_discount) / max_abs_rew_discount))

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
            print("t {:.2f} steps {:6} reward {:.2f} reward disc {:.2f} avg {:.2f}, avg100 {:.2f}, eps {:.3f} " \
                    "updates {:8} total steps {:8} epoch {:.1f} lr {:.5f}".format((time.time() - startt) / steps * 100.,\
                    steps, total_rew / agent.config['scalereward'],total_rew_discount / agent.config['scalereward'],
                    avg / agent.config['scalereward'],np.mean(np.array(totrewlist[-100:])),\
                    agent.epsilon(eps), agent.config['num_updates'],total_steps, agent.config['num_updates']/50000, agent.getlearnrate()))
        if params['monitor'] == True:
            env.monitor.close()
        print(agent.config)

        #gym.upload(resultsdir, writeup='https://gist.github.com/davidenitti/c6cd38cd1151ccc0248da8fe1f78e6ee', api_key='')# gym.upload(resultsdir, api_key='YOURAPI')
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
