import numpy as np
import time
import cv2
import threading
import random
import logging
logger = logging.getLogger(__name__)

def onehot(i, n, dtype=np.float32):
    out = np.zeros(n,dtype=dtype)
    out[i] = 1.
    return out


def do_rollout(agent, env, episode, num_steps=None, render=False, useConv=True, discount=1,
               learn=True, sleep=0.):
    if num_steps == None:
        num_steps = env.spec.max_episode_steps
    total_rew = 0.
    total_rew_discount = 0.
    cost = 0.
    if 'scaling' in agent.config:
        scaling = agent.config['scaling']
    else:
        scaling = 'none'
    ob = env.reset()
    ob = preprocess(ob, agent.observation_space, agent.scaled_obs, type=scaling)

    if agent.config['terminal_life']:
        last_lives = -1

    if useConv == False:
        ob = ob.reshape(-1, )
    ob1 = np.copy(ob)
    for _ in range(agent.config["past"]):
        ob1 = np.concatenate((ob1, ob), 0)

    if 'transition_net' in agent.config and agent.config['transition_net']: #fixme render and
        agent.state_list=[[],[]]
        update_state = True
    else:
        update_state = False

    min_reward=float("inf")
    max_reward = float("-inf")
    max_qval = float("-inf")
    for t in range(num_steps):
        if sleep > 0:
            time.sleep(sleep)
        if agent.config['policy']:
            a = agent.actpolicy(ob1, episode)
        else:
            a = agent.act(ob1, episode,update_state=update_state)

        start_time = time.time()
        (obnew, rr, done, _info) = env.step(a)
        start_time2 = time.time()
        if agent.config['terminal_life']:
            if _info['ale.lives'] < last_lives:
                terminal_memory = True
            else:
                terminal_memory = done
            last_lives = _info['ale.lives']
        else:
            terminal_memory = done

        #print(reward, done,terminal_memory, _info['ale.lives'])
        obnew = preprocess(obnew, agent.observation_space, agent.scaled_obs, type=scaling)

        min_reward = min(min_reward,rr)
        max_reward = max(max_reward, rr)

        reward = rr*agent.config['scalereward']
        if agent.config['limitreward'] is not None:
            limitreward = min(agent.config['limitreward'][1], max(agent.config['limitreward'][0], reward))
        else:
            limitreward = reward

        if useConv == False:
            obnew = obnew.reshape(-1, )

        if len(ob.shape) == 3:
            obnew1 = np.concatenate((ob1[obnew.shape[0]:, :, :], obnew), 0)
        else:
            obnew1 = np.concatenate((ob1[ob.shape[0]:], obnew))

        #old
        #agent.memory.add([ob1, a, limitreward, 1. - 1. * terminal_memory, t, None])
        agent.memory.add(ob, a, limitreward, 1. - 1. * terminal_memory, t, np.nan)
        start_time3 = time.time()
        if learn and (not agent.config['policy']):
            cost += agent.learn()
        elif ((t + 1) % agent.config['batch_size'] == 0 or done) and agent.config['policy']:
            # raise NotImplemented("startind not good for circular buffer")
            agent.learnpolicy()
            agent.memory.empty()

        total_rew_discount += limitreward * (discount ** t) #using limited reward
        total_rew += reward

        if (t % 200 == 0 or done):
            if agent.config['policy']:
                logger.debug("{} episode {} step {} done {} V {} limitedrew {}".format(agent.config["path_exp"], episode, t, done, agent.evalV(ob1[None,...], True), limitreward))
            else:
                q_val = agent.evalQ(ob1[None,...])
                max_qval = max(max_qval,np.max(q_val))
                logger.debug("{} episode {} step {} done {} Q {} limitedrew {}".format(agent.config["path_exp"], episode, t, done, 'Q', q_val, limitreward))

        ob1 = obnew1
        ob = obnew

        start_time4 = time.time()
        if render and t % 3 == 0:  # render every X steps (X=1)
            env.render()

        if done: break
        if t % 5 == 0 and False:#fixme
            print("time step",time.time()-start_time, start_time2 - start_time,start_time3 - start_time2,
                  start_time4-start_time3,time.time() - start_time4)

    return total_rew, t + 1, total_rew_discount, max_qval


def preprocess(observation, observation_space, scaled_obs, type='none'):
    if type == 'crop':
        resize_height = int(round(
            float(observation.shape[0]) * scaled_obs[1] / observation.shape[1]))
        observation = cv2.cvtColor(
            cv2.resize(observation, (scaled_obs[1], resize_height), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
        crop_y_cutoff = resize_height - 8 - scaled_obs[0]
        cropped = observation[crop_y_cutoff:crop_y_cutoff + scaled_obs[0], :]
        return cropped[None,...]#np.reshape(cropped, scaled_obs)
    elif type == 'scale':
        return cv2.cvtColor(cv2.resize(observation, (scaled_obs[1], scaled_obs[0]), interpolation=cv2.INTER_LINEAR),
                            cv2.COLOR_BGR2GRAY)[None,...]
    elif type == 'none' or np.isinf(observation_space.low).any() or np.isinf(observation_space.high).any():
        return observation
    elif type == 'flat':
        o = (observation - observation_space.low) / (observation_space.high - observation_space.low) * 2. - 1.
        return o.reshape(-1, )




def vis(pl, w, image, images1, images2):
    channels=1
    pl.clf()
    cols = max(w[0].shape[0], images2.shape[0])
    for i in range(w[0].shape[0]):
        pl.subplot(9, cols, 1 + i + cols * 6)
        img = np.concatenate((w[0][i, :channels, :, :], w[0][i, channels:channels*2,:, :]),2)

        img=img[0]#.reshape(img.shape[1],img.shape[2])
        pl.imshow((img - img.min()) / (img.max() - img.min() + 1e-20), vmin=0, vmax=1, cmap=pl.get_cmap('gray'))
        pl.axis('off')
    if len(w)>1:
        for i in range(w[1].shape[0]):
            pl.subplot(9, cols, 1 + i + cols * 7)
            img = np.concatenate((w[1][i, :channels, :, :], w[1][i, channels:channels*2,:, :]),2)
            img = img[0]
            pl.imshow((img - img.min()) / (img.max() - img.min() + 1e-20), vmin=0, vmax=1, cmap=pl.get_cmap('gray'))
            pl.axis('off')

    for i in range(images1.shape[0]):
        pl.subplot(4, images1.shape[2] + 1, i + 1)
        img = images1[i,:, :].reshape(-1, images1.shape[1])
        # print img.shape,images2.shape
        pl.imshow((img - img.min()) / (img.max() - img.min() + 1e-20), vmin=0, vmax=1, cmap=pl.get_cmap('gray'))
        pl.axis('off')
        #print('filtered 1', img.max(), img.min())
    for i in range(images2.shape[0]):
        #print(images1.shape[2] + 1, i + 2 + images1.shape[2])
        pl.subplot(4, images1.shape[2] + 1, i + 2 + images1.shape[2])
        img = images2[i, :, :].reshape(-1, images2.shape[1])
        # print img.shape,images2.shape
        pl.imshow((img - img.min()) / (img.max() - img.min() + 1e-20), vmin=0, vmax=1, cmap=pl.get_cmap('gray'))
        pl.axis('off')
        #print('filtered 2',img.max(), img.min())

    pl.subplot(4, images1.shape[0] + 1, 1 + 1 + images1.shape[0] + images2.shape[0])
    img = image[0, :, :]
    #print('img',image.min(), image.max())
    for aa in range(1, image.shape[0]):
        img = np.concatenate((img, image[aa, :, :]),1)  # image.reshape(-1,image.shape[1],3)
    pl.imshow((img - img.min()) / (img.max() - img.min() + 1e-20), vmin=0, vmax=1, cmap=pl.get_cmap('gray'))
    pl.axis('off')


