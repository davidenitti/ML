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
import argparse
import os
import pickle

def loadparams(filename):
    with open(filename + ".p", "rb") as input_file:
        out=pickle.load(input_file)
    return out

def getparams():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default="")
    args = parser.parse_args()

    if args.file!="" and os.path.isfile(args.file + ".p"):
        prms=loadparams(args.file)
        options=prms[-1]
    else:
        options={
            'target':"LunarLander-v2",
            'episodes':1000000,
            'render':True,
            'plot':True,
            'monitor':False,
            'logging':1
        }

    return options

def main(numavg=100):
    options = getparams()

    if options['plot'] != True:
        import matplotlib
        matplotlib.use('pdf')
    else:
        import matplotlib.pyplot as plt
        plt.rcParams['image.interpolation'] = 'nearest'

    nameenv = options['target']

    reward_threshold = gym.envs.registry.spec(nameenv).reward_threshold
    env = gym.make(nameenv)

    resultsdir = "./data" + nameenv

    if options['monitor'] == True: # store performance and video (check https://gym.openai.com/docs )
        from gym import wrappers
        env = wrappers.Monitor(env, resultsdir, force=True)

    params = default_params.get_default(nameenv)
    params.update(options)
    logger = common.init_log(params)

    logger.info(str((env.observation_space, env.action_space, env.spec.timestep_limit, env.reward_range,
                     gym.envs.registry.spec(nameenv).trials)))
    for p in params:
        logger.debug(p+" "+str(params[p]))

    if params["seed"] >0:
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
        stepslist = []
        greedyrewlist = [[],[]]
        totrewavglist = []
        total_rew_discountlist = []
        if params['plot'] == True:
            showevery = 10
        else:
            showevery = 100
        testevery = 20
        useConv=agent.config['conv']

        print(agent.config)

        agent.start_async_plot([(totrewlist,totrewavglist,greedyrewlist),reward_threshold], plt,params['plot'] == True)

        if agent.config['threads'] > 0:
            agent.start_async_learning(num_threads=None,delay=agent.config['delay_threads'])

        for episode in range(params['episodes']):
            time.sleep(agent.config['delay_threads'])
            if agent.config['copyQ'] >0 and (episode + 1) % agent.config['copyQ'] == 0:
                agent.copyq()
                print("coping Q...")

            if (episode) % testevery == 0 or episode>=params['episodes']-numavg:
                render = (params['render'])
                eps = -1
                learn=False
                print(agent.config['file'],'episode', episode, 'l rate', agent.getlearnrate(),'lambda',agent.config['lambda'])
            else:
                render = False
                learn = True
                eps = episode
            startt = time.time()
            total_rew, steps, total_rew_discount = torchagent.do_rollout(agent, env, eps, num_steps=num_steps,
                            render=render, diff=agent.config["diffstate"], useConv=useConv,
                            actevery=agent.config["actevery"], discount=agent.config["discount"],
                            sleep=0.0, learn=learn)

            if (episode) % testevery == 0:
                if avg is None:
                    avg = total_rew
                greedyrewlist[0].append(total_rew / agent.config['scalereward'])
                greedyrewlist[1].append(episode)
                inc = max(0.2,0.05+ 1. / (episode + 1.) ** 0.5)
                avg = avg * (1 - inc) + inc * total_rew
                totrewavglist.append(avg / agent.config['scalereward'])

            if episode % 10 == 0:
                print(agent.config)
            if episode % 100 == 0 and episode>0:
                if agent.config['file'] is not None:
                    print("saving...")
                    agent.save()
            if episode % 1 == 0:

                totrewlist.append(total_rew / agent.config['scalereward'])

                if (episode + 1) % showevery == 0:
                    stepslist.append(steps)
                total_rew_discountlist.append(total_rew_discount / agent.config['scalereward'])

            print("time {:.3f} steps {:6} total reward {:4} total reward disc {:.3f} avg {:.3f}, eps {:.3f} " \
                    "updates {:8} epoch {:.1f} lr {:.5f}".format((time.time() - startt) / steps * 100.,\
                    steps, total_rew / agent.config['scalereward'],total_rew_discount / agent.config['scalereward'],
                    avg / agent.config['scalereward'],\
                    agent.epsilon(eps), agent.numupdates, agent.numupdates/50000, agent.getlearnrate()))
        if params['monitor'] == True:
            env.monitor.close()
        print(agent.config)

        #gym.upload(resultsdir, writeup='https://gist.github.com/davidenitti/c6cd38cd1151ccc0248da8fe1f78e6ee', api_key='sk_7mpGWQjpTmWMv4ym739Xg')# gym.upload(resultsdir, api_key='YOURAPI')
    except Exception as e:
        print('Exception',e)
        agent.coord.request_stop(e)
        agent.stop_threads()
        raise e
    agent.stop_threads()
    agent.close()
    return np.mean(totrewlist[-numavg:]),agent.config,totrewlist,totrewavglist,greedyrewlist,reward_threshold

if __name__ == '__main__':
    main()