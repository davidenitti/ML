import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os,shutil
import gym
import time
import threading
import numpy as np
import common,cv2
import torch.optim as optim
import tensorflow as tf
from torch.autograd import Variable
def onehot(i, n):
    out = np.zeros(n)
    out[i] = 1.
    return out

def do_rollout(agent, env, episode, num_steps=None, render=False, diff=True, useConv=True, actevery=1, discount=1,
               learn=True, sleep=0.):
    if num_steps == None:
        num_steps = env.spec.timestep_limit
    total_rew = 0.
    total_rew_discount = 0.
    cost = 0.
    if 'scaling' in agent.config:
        scaling=agent.config['scaling']
    else:
        scaling='none'
    ob = env.reset()
    ob = preprocess(ob,agent.observation_space,agent.scaled_obs,type=scaling)

    if useConv == False:
        ob = ob.reshape(-1, )
    ob1 = np.copy(ob)
    for _ in range(agent.config["past"]):
        ob1 = np.concatenate((ob1, ob), len(ob.shape) - 1)
    #agent.memoryLock.acquire()
    #startind=agent.sizemem
    #agent.memoryLock.release()
    for t in range(int(num_steps/actevery)):
        start_time = time.time()
        if agent.config['policy']:
            a,wa = agent.actpolicy(ob1, episode)
        else:
            a = agent.act(ob1, episode)
            wa = 1.

        done=False
        reward = 0.
        listcurobs=[]
        for ii in range(actevery):
            if not done:
                (obnew, rr, done, _info) = env.step(a)
                obnew = preprocess(obnew,agent.observation_space,agent.scaled_obs,type=scaling)
                rr *= agent.config['scalereward']
                reward += rr
                listcurobs.append(obnew) # to remove
        obnew=np.concatenate(listcurobs, len(ob.shape) - 1)
        if agent.config['limitreward'] is not None:
            limitreward = min(agent.config['limitreward'][1],max(agent.config['limitreward'][0],reward))
        else:
            limitreward = reward

        if useConv == False:
            obnew = obnew.reshape(-1, )

        if len(ob.shape) == 3:
            obnew1 = np.concatenate((ob1[:, :, obnew.shape[-1]:], obnew), len(ob.shape) - 1)
        else:
            obnew1 = np.concatenate((ob1[ob.shape[0]:], obnew))
        agent.memory.add([ob1, a, limitreward, 1. - 1. * done, t, wa,None])

        if agent.config['threads']==0:
            if agent.newpolicy:
                cost += agent.learn()
            if learn and (not agent.config['policy']):
                cost += agent.learn()
            elif ((t+1) % agent.config['batch_size'] == 0 or done) and agent.config['policy'] and agent.config['threads'] == 0 and episode is not None and episode >= 0:
                    raise NotImplemented("startind not good for circular buffer")
                    agent.learnpolicy(startind, agent.sizemem)#fixme
                    agent.memoryLock.acquire()
                    startind = agent.sizemem#fixme
                    agent.memoryLock.release()

        ob1 = obnew1
        ob = obnew
        total_rew_discount += reward * (discount ** t)
        total_rew += reward

        if (t % 200 == 0 or done):
            if agent.config['policy']:
                print(agent.config['file'], 'episode', episode, t, done, 'V', agent.sess.run(agent.V, feed_dict={
                    agent.x: ob1.reshape(tuple([1] + list(ob1.shape)))}), reward)
            else:
                print(agent.config['file'], 'episode', episode, t, done, 'Q',
                      agent.evalQ(ob1.reshape(tuple([1] + list(ob1.shape)))), reward)

        if render and t % 1 == 0: # render every X steps (X=1)
            env.render()
            if sleep > 0:
                time.sleep(sleep)
        if done: break
        if t%100==0:
            agent.learnrate *= agent.config['decay_learnrate']
        if t%100==0:
            print("time step",time.time()-start_time)
    adjust_learning_rate(agent.optimizer,agent.learnrate)
    return total_rew, t + 1, total_rew_discount

def preprocess(observation,observation_space,scaled_obs,type='none'):
    if type=='crop':
        resize_height = int(round(
            float(observation.shape[0]) * scaled_obs[1] / observation.shape[1]))
        observation = cv2.cvtColor(cv2.resize(observation, (scaled_obs[1], resize_height),interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
        crop_y_cutoff = resize_height - 8 - scaled_obs[0]
        cropped = observation[crop_y_cutoff:crop_y_cutoff + scaled_obs[0], :]
        return np.reshape(cropped,scaled_obs)
    elif type=='scale':
        return cv2.cvtColor(cv2.resize(observation, (scaled_obs[1],scaled_obs[0]), interpolation=cv2.INTER_LINEAR),
                                   cv2.COLOR_BGR2GRAY).reshape(scaled_obs)
    elif type=='none' or np.isinf(observation_space.low).any() or np.isinf(observation_space.high).any():
        return observation
    elif type == 'flat':
        o = (observation-observation_space.low) / (observation_space.high - observation_space.low) * 2. - 1.
        return o.reshape(-1,)

def identity(inp):
    return inp

def activ(activation):
    if activation == 'lrelu':
        return F.leaky_relu
    elif activation == 'relu':
        return F.relu
    elif activation == 'tanh':
        return F.tanh
    elif activation == 'elu':
        return F.elu
    elif activation == None:
        return identity
    else:
        raise Exception("activation "+activation+" not supported")

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class ConvNet(nn.Module):
    def __init__(self,convlayers,input_channel=1,scale=1.0,final_act=False,activation='relu',batch_norm=False):
        super(ConvNet, self).__init__()
        self.conv = []
        for i in range(len(convlayers)):
            self.conv.append(nn.Conv2d(input_channel, convlayers[i][3],
                                       kernel_size=convlayers[i][0], stride=convlayers[i][2]))
        self.conv = nn.ModuleList(self.conv)
        self.scale = scale
        self.act = activ(activation)
    def forward(self, x):
        x = x * self.scale
        for c in self.conv[:-1]:
            x = c(x)
            x = self.act(x)
        x = self.conv[-1](x)
        if self.final_act:
            x = self.act(x)
        x = x.view(-1, 320)  # fixme
        return x

class DenseNet(nn.Module):
    def __init__(self,dense_layers,input_shape,scale=1.0,final_act=False,activation='relu',batch_norm=False):
        super(DenseNet, self).__init__()
        self.dense = []

        self.dense_layers = [input_shape]+dense_layers
        if batch_norm:
            self.bn = [nn.BatchNorm1d(self.dense_layers[0])]
        for i in range(len(self.dense_layers)-1):
            self.dense.append(nn.Linear(self.dense_layers[i], self.dense_layers[i+1]))
            if batch_norm:
                self.bn.append(nn.BatchNorm1d(self.dense_layers[i+1]))
        self.dense = nn.ModuleList(self.dense)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.ModuleList(self.bn)
        self.scale = scale
        self.final_act = final_act
        self.act =  activ(activation)
    def forward(self, x):
        x = x*self.scale
        if self.batch_norm:
            x = self.bn[0](x)
        for i,d in enumerate(self.dense[:-1]):
            x = d(x)
            if self.batch_norm:
                x = self.bn[i+1](x)
            x = self.act(x)
        x = self.dense[-1](x)
        if self.final_act:
            if self.batch_norm:
                x = self.bn[-1](x)
            x = self.act(x)
        return x #F.log_softmax(x, dim=1)

class Memory(object):
    def __init__(self,max_size=100000,copy=False):
        self.mem = []
        self.max_size = max_size
        self.memoryLock = threading.Lock()
        self.last_ind = -1
        self.start_ind = -1
        self.copy = copy
    def __getitem__(self, item): # item as to be from 0 to len(mem)-1
        assert (item<len(self.mem))
        idx = (self.start_ind+item) % self.max_size
        if self.copy:
            self.memoryLock.acquire()
            val = self.mem[idx].deepcopy()
            self.memoryLock.release()
        else:
            val = self.mem[idx]
        return val
    def add(self,example):
        self.memoryLock.acquire()
        self.last_ind += 1
        self.last_ind = self.last_ind % self.max_size
        if len(self.mem)>=self.max_size:
            self.mem[self.last_ind]=example
            self.start_ind = (self.last_ind+1) % self.max_size
        else:
            self.mem.append(example)
            self.start_ind = 0
        self.memoryLock.release()
    def sizemem(self):
        #self.memoryLock.acquire()
        l = len(self.mem)
        #self.memoryLock.release()
        return l

class deepQconv(object):
    def __del__(self):
        self.close()

    def save(self, filename=None):
        if filename is None:
            filename = self.config['file']
        with open(filename + ".p", "wb") as input_file:
            pickle.dump((self.observation_space, self.action_space, self.reward_range, self.config), input_file)
        save_path = self.saver.save(self.sess, os.path.abspath(filename + ".tf"))

    def __init__(self,logger, observation_space, action_space, reward_range, **userconfig):
        self.logger=logger

        if userconfig["file"] is not None:
            self.log_dir = userconfig["file"]
            print(self.log_dir)
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir)
            os.makedirs(self.log_dir)
        else:
            self.log_dir = None

        if userconfig["file"] is not None and os.path.isfile(userconfig["file"] + ".p"):
            with open(userconfig["file"] + ".p", "rb") as input_file:
                self.observation_space, self.action_space, self.reward_range, self.config = pickle.load(input_file)

        else:
            self.observation_space = observation_space
            print(self.observation_space)
            self.action_space = action_space
            self.reward_range = reward_range
            self.config = userconfig

        self.isdiscrete = isinstance(self.action_space, gym.spaces.Discrete)
        print(self.action_space)
        print(self.config)
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise Exception('Observation space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(
                observation_space, self))
        self.learnrate = self.config['initial_learnrate']
        self.initQnetwork()

    def learn_thread(self,force=True,print_it=True,delay=0.00000001,policygrad=False):
        try:
            while not self.coord.should_stop():
                if policygrad:
                    #raise Exception("error policy not implemented")
                    self.learnpolicy()
                else:
                    self.learn(force, print_it)
                if self.newpolicy:
                    self.learn(force, print_it)
                time.sleep(delay)
        except Exception as e:
            print('Exception',e)
            self.coord.request_stop(e)
            raise e


    def start_async_learning(self,num_threads=None,delay=0.00000001):
        if num_threads is None:
            num_threads = self.config['threads']
        for _ in range(num_threads):
            ln = threading.Thread(target=self.learn_thread, args=(True,True,delay,self.config['policy']))
            ln.start()
            self.workers.append(ln)
    def start_async_plot(self, arg, plt, winplot,numplot=0):
        lists, reward_threshold = arg
        if self.useConv:
            with tf.variable_scope("shared", reuse=True):
                with tf.variable_scope("conv0", reuse=True):
                    ww = [self.getweights2('w')]  # to fix!

            with tf.variable_scope("shared", reuse=True):
                with tf.variable_scope("conv1", reuse=True):
                    ww += [self.getweights2('w')]
        else:
            ww=[]
        t = threading.Thread(target=self.loopplot, args=(ww,lists, reward_threshold, plt, winplot,numplot))
        #t = multiprocessing.Process(target=self.processPlot, args=(arg, plt, winplot))
        t.start()
        self.workers.append(t)
    def stop_threads(self):
        self.coord.request_stop()
        for t in self.workers:
            t.join()
        self.close()

    def loopplot(self,ww, lists,reward_threshold, plt, plot=True,numplot=0):
        totrewlist,totrewavglist,greedyrewlist = lists
        while not self.coord.should_stop():
            if len(totrewlist) > 10:
                self.plot(ww,lists, reward_threshold, plt, plot=plot, numplot=numplot)
                if plot==False:
                    for _ in range(60):
                        if not self.coord.should_stop():
                            time.sleep(2.)
                        else:
                            break
            else:
                if plot:
                    time.sleep(5.)
                else:
                    time.sleep(1.)
    def plot(self, w, lists,reward_threshold, plt, plot=True,numplot=0):
        width = 15000
        totrewlist,totrewavglist,greedyrewlist = lists
        if len(totrewlist) > 10:
            if plot:
                fig = plt.figure(numplot)
                fig.canvas.set_window_title(str(self.config["file"]) + " " + str(self.config))
            plt.clf()
            plt.xlim(max(-1, len(totrewlist) / 50 * 50 - width), len(totrewlist) // 50 * 50 + 50)
            plt.ylim(min(totrewlist[max(0, len(totrewlist) // 50 * 50 - width):]) - 5,
                     max(totrewlist[max(0, len(totrewlist) // 50 * 50 - width):]) + 5)
            plt.plot([0, len(totrewlist) + 100], [reward_threshold, reward_threshold], color='green')

            plt.plot(range(len(totrewlist)), totrewlist, color='red')
            # plt.plot(range(len(total_rew_discountlist)), [ddd-total_rew_discountlist[0]+totrewlist[0] for ddd in total_rew_discountlist], color='green')
            plt.plot(greedyrewlist[1], totrewavglist, color='blue')
            plt.scatter(greedyrewlist[1], greedyrewlist[0], color='black')
            plt.plot([max(0, len(totrewlist) - 1 - 100), len(totrewlist) - 1],
                     [np.mean(np.array(totrewlist[-100:])), np.mean(np.array(totrewlist[-100:]))], color='black')
            if self.config["file"] is not None:
                plt.savefig(self.config["file"] + '_reward' +  '.png', dpi=400)
            #else:
            #    plt.savefig('/tmp/reward_temp.png', dpi=200)
            # if args.plot.lower() == "true":
            #    fig = plt.figure(1)
            #    fig.canvas.set_window_title(str(agent.config["file"]) + " " + str(agent.config))
            # plt.clf()
            # plt.plot(range(len(stepslist)), stepslist, color='black')
            # if agent.config["file"] is not None:
            #    plt.savefig('steps_'+agent.config["file"]+'.png', dpi=500)
            if self.useConv: # to fix!
                if plot:
                    fig = plt.figure(2)
                    fig.canvas.set_window_title(str(self.config["file"]) + " " + str(self.config))

                self.memoryLock.acquire()
                if self.memory.sizemem() > 1:
                    selec = np.random.choice(self.memory.sizemem()-1)
                    imageselected = self.memory[selec][0].copy()
                    self.memoryLock.release()
                    print('unlock plot')
                    filtered1 = self.sess.run(self.convout[0], feed_dict= \
                        {self.x: imageselected.reshape(tuple([1] + list(imageselected.shape)))})
                    if len(self.convout)>1:
                        filtered2 = self.sess.run(self.convout[1], feed_dict= \
                            {self.x: imageselected.reshape(tuple([1] + list(imageselected.shape)))})
                    else:
                        filtered2 = filtered1
                    print(imageselected.shape, imageselected.min(), imageselected.max(),
                          filtered1[0].shape, filtered2[0].shape)
                    vis(plt, [self.sess.run(www) for www in w], imageselected, filtered1[0], filtered2[0])

                    if self.config["file"] is not None and plot==False:
                        plt.savefig(self.config["file"] + '_features' + '.png', dpi=300)
                    if self.summary_writer is not None:
                        # nextstate = state tofix!
                        summary_str = self.sess.run(self.summary,
                                feed_dict={self.nextstate: imageselected.reshape(tuple([1] + list(imageselected.shape))),self.x: imageselected.reshape(tuple([1] + list(imageselected.shape)))})

                        self.summary_writer.add_summary(summary_str, 1)
                        self.summary_writer.flush()
                else:
                    self.memoryLock.release()
            else:
                self.memoryLock.acquire()
                if self.memory.sizemem():
                    selec = np.random.choice(self.memory.sizemem()-1)
                    stateelected = self.memory[selec][0]#.copy()
                    # todo add summary
                self.memoryLock.release()

            if plot:
                plt.draw()
                plt.pause(.5)
                time.sleep(1.0)

    def stateautoencoder(self, n_input,layers):

        self.outstate, regul, self.hiddencode = multilayer_perceptron(self.x, [n_input]+layers+[n_input],
                                                                      [0.000001]*(len(layers)+1), outhidden=1)
        # print self.x.get_shape(),self.outstate.get_shape()
        self.costautoenc = tf.reduce_mean((self.sa - self.outstate) ** 2) + regul
        self.optimizerautoenc = tf.train.RMSPropOptimizer(self.learnrate, 0.99,0.0,1e-6).minimize(self.costautoenc,
                                                                                              global_step=self.global_step)

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
            return max(self.config['mineps'],self.config['eps'] * self.config['decay'] ** episode)

    def getlearnrate(self):
        return self.learnrate
    def commoninit(self):
        self.newpolicy=False
        chk,msg=common.checkparams(self.config)
        if chk==False:
            raise Exception(msg)

        self.newQ=True

        if 'convbias' not in self.config:
            self.config['convbias']=0.0

        self.fulldouble = self.config['doubleQ'] and self.config['copyQ'] <= 0
        self.copyQalone = self.config['copyQ'] > 0 and self.config['doubleQ'] == False

        self.useConv = self.config['conv']
        if 'dimdobs' in self.config:
            self.scaled_obs = self.config['dimdobs']
        else:
            self.scaled_obs = self.observation_space.shape

    def sharednet(self,input):
        convout = None
        if self.useConv:
                convout, dense = convlayer(input, activation=self.config['activation'],
                                                     numchannels=1 + 1 * self.config['past'],
                                                     convlayers=self.config['convlayers'],
                                                     scaleobs=self.config['scaleobs'],
                                                     type_initializer=self.config['initializer'])
                if self.config['shareallnet']:
                    dense = MLP(dense, self.config['hiddenlayers'], self.config['regularization'],
                           scaleobs=1, activation=self.config['activation'],
                            activationout=self.config['activation'], addsummary=True,type_initializer=self.config['initializer'])

        else:
            if self.config['shareallnet']:
                dense = DenseNet(self.config['hiddenlayers'],input_shape=input[-1],
                                 scale=self.config['scaleobs'],final_act=True,
                                 activation=self.config['activation'], batch_norm=self.config["batch_norm"])
                # dense = MLP(input, self.config['hiddenlayers'], self.config['regularization'],
                #            scaleobs=self.config['scaleobs'], activation=self.config['activation'],
                #             activationout=self.config['activation'],addsummary=True,type_initializer=self.config['initializer'])
            else:
                raise NotImplemented
                dense = tf.to_float(input) * self.config['scaleobs']
        return convout,dense

    def Qnet(self, input):
        if self.config['shareallnet']:
            layers=[]
        else:
            layers=self.config['hiddenlayers']
        Q = DenseNet(layers + [self.n_out],input_shape=input,scale=1,
                     activation=self.config['activation'], batch_norm=self.config["batch_norm"])
            #MLP(input, layers + [self.n_out], self.config['regularization'],
            #scaleobs=1, activation=self.config['activation'],type_initializer=self.config['initializer'],
            #    activationout=None, addsummary=True)
        return Q
    def fullQNet(self,input,reuse):
        with tf.variable_scope('shared',reuse=reuse):
            _, dense2 = self.sharednet(input)
        with tf.variable_scope('Q',reuse=reuse):
            QQ = self.Qnet(dense2)
        return QQ
    def policyNet(self, input):
        if self.config['shareallnet']:
            layers=[]
        else:
            layers=self.config['hiddenlayers']
        logitpolicy = MLP(input, layers + [self.n_out], self.config['regularization'],
            scaleobs=1, activation=self.config['activation'], type_initializer=self.config['initializer'],
                          activationout=None, addsummary=True)
        policy = tf.nn.softmax(logitpolicy)
        return policy,logitpolicy

    def Vnet(self, input):
        if self.config['shareallnet']:
            layers=[]
        else:
            layers=self.config['hiddenlayers']
        V = MLP(input, layers+[1], self.config['regularization'],
                    scaleobs=1, activation=self.config['activation'],type_initializer=self.config['initializer'],
                activationout=None, addsummary=True)
        return V


    def initQnetwork(self):
        self.commoninit()
        use_cuda = True
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.device_eval = torch.device("cpu")
        if self.isdiscrete:
            print('scaled',self.scaled_obs)
            n_input = self.scaled_obs[:-1] + (self.scaled_obs[-1] * (self.config['past'] + 1),)  # + self.observation_space.shape[0]*onehot(0,len(self.observation_space.shape))*(self.config['past'])
            if self.useConv == False:
                n_input = [int(np.prod(self.scaled_obs + (1 * (self.config['past'] + 1),)))]
            self.n_out = self.action_space.n
            print('n_input',n_input,self.observation_space.shape,self.scaled_obs)

        print(self.observation_space,'obs', n_input, 'action', self.n_out)

        self.convout, self.dense = self.sharednet(n_input)


        if self.config['policy']:
            with tf.variable_scope('policy'):
                self.policy, self.logitpolicy = self.policyNet(self.dense)
            with tf.variable_scope('V'):
                self.V = self.Vnet(self.dense)
        if self.newpolicy or self.config['policy'] == False:
            self.Q = nn.Sequential(self.dense,self.Qnet(self.dense.dense_layers[-1]))
            self.Q = self.Q.to(self.device)
            # this used a shared net for copyQ or doubleQ
            if self.config['copyQ'] > 0 or self.config['doubleQ']:
                raise NotImplemented
                with tf.variable_scope('copyQ'):
                    self.Q2 = self.fullQNet(self.x,reuse=False)
                self.copyQ2_Q = make_copy_params_op(
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/copyQ/Q'),
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/Q'))
                copyQ2_Q2 = make_copy_params_op(
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name + '/copyQ/shared'),
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name + '/shared'))
                self.copyQ2_Q = self.copyQ2_Q+copyQ2_Q2

        self.criterion = nn.SmoothL1Loss()
        if self.config['policy']:
            #mean, var = tf.nn.moments(self.yp,axes=[0])
            #self.adv=self.yp #tf.nn.batch_normalization(self.yp, adv_mean, adv_var, 0, 1, 0.0000001, 'batchnormadv')
            deltaV =  self.y - self.V
            print('shapeV y', self.V.get_shape(), self.y.get_shape())
            # assert self.V.get_shape() == self.y.get_shape()
            if self.config['clip']>0 and "cliptype" in self.config and self.config['cliptype']=='deltaclip':
                self.errorlistV = tf.reduce_mean(clipdelta(deltaV,self.config['clip']))
            else:
                self.errorlistV = tf.reduce_mean(0.5 * deltaV ** 2)
            if self.config['entropy']<=0.0:
                entropy=0.0
            else:
                entropy = self.config['entropy'] * tf.reduce_mean(-tf.reduce_sum(self.policy * tf.nn.log_softmax(self.logitpolicy), 1))
            #print('sh',(self.yp * tf.nn.softmax_cross_entropy_with_logits(labels=self.curraction,logits=self.logitpolicy)).get_shape())
            self.errorpolicy =  \
                tf.reduce_mean(self.yp * tf.nn.softmax_cross_entropy_with_logits(labels=self.curraction,logits=self.logitpolicy) )\
                -entropy

            self.errorpolicy = self.errorlistV + self.errorpolicy
            regul = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,tf.get_variable_scope().name)
            printRegularization()
            if regul != []:
                self.errorpolicy+=tf.add_n(regul)
            '''
            self.optimizer_policy = tf.train.RMSPropOptimizer(self.learnrate, decay=self.config['decayoptimizer'],
                        momentum=self.config['momentum'], epsilon=self.config['epsoptimizer'])
            self.optimizer_V = 0.0
            if self.config["clip"] > 0:
                self.optimizer_policy = clipgrad(self.optimizer_policy, self.errorpolicy,
                                                 global_step=self.global_step, lim=self.config['clip'])
            else:
                self.optimizer_policy = self.optimizer_policy.minimize(self.errorpolicy,
                                                                       global_step=self.global_step)
            '''
            self.optimizer_policy = tf.train.RMSPropOptimizer(self.learnrate, self.config['decayoptimizer'],
                                                              self.config['momentum'], epsilon=self.config['epsoptimizer'])
            #self.optimizer_V = tf.train.RMSPropOptimizer(self.learnrate, self.config['decayoptimizer'],
            #                                            self.config['momentum'], epsilon=self.config['epsoptimizer'])
            if self.config["clip"]>0 and self.config['cliptype']!='deltaclip':
                self.optimizer_policy = self.clip(self.optimizer_policy, self.errorpolicy, global_step=self.global_step,lim=self.config['clip'])
                #self.optimizer_V = self.clip(self.optimizer_V, self.errorlistV + regul_V, global_step=self.global_step,lim=self.config['clip'])
            else:
                self.optimizer_policy = self.optimizer_policy.minimize(self.errorpolicy, global_step=self.global_step)
                #self.optimizer_V = self.optimizer_V.minimize(
                #                            self.errorlistV + regul_V, global_step=self.global_step)
        if self.newpolicy or self.config['policy']==False:
            #self.singleQ = tf.reduce_sum(self.curraction * self.Q, reduction_indices=1)  # tf.reduce_sum(input_tensor, reduction_indices, keep_dims, name)#tf.slice(self.Q, [0,self.curraction],[-1,1])  #self.Q[:,self.out]
            #print('shapeQ y', self.curraction.get_shape(), self.Q.get_shape(), self.singleQ.get_shape(), self.y.get_shape())

            #self.Qnext = self.fullQNet(self.nextstate,reuse=True)
            if self.copyQalone:
                raise NotImplemented

            # if self.copyQalone:
            #     with tf.variable_scope('copyQ', reuse=True):
            #         self.Q2next = self.fullQNet(self.nextstate, reuse=True)
            #     #with tf.variable_scope('Q2',reuse=True):
            #     #    self.Q2next = self.Qnet(densenext)
            #     maxQnext = tf.reduce_max(self.Q2next, reduction_indices=1)
            # else:
            #     maxQnext = tf.reduce_max(self.Qnext,reduction_indices=1)
            # print('maxQnext',maxQnext.get_shape())
            # if self.config['episodic']:
            #     delta = self.singleQ - (self.reward+self.config["discount"]*tf.stop_gradient(maxQnext)*self.notdone)
            # else:
            #     delta = self.singleQ - (self.reward + self.config["discount"] * tf.stop_gradient(maxQnext))
            #

            # assert self.singleQ.get_shape() == self.y.get_shape()
            if self.config['doubleQ']:
                self.singleQ2 = tf.reduce_sum(self.curraction * self.Q2,
                                              reduction_indices=1)  # tf.reduce_sum(input_tensor, reduction_indices, keep_dims, name)#tf.slice(self.Q, [0,self.curraction],[-1,1])  #self.Q[:,self.out]
                with tf.variable_scope('copyQ', reuse=True):
                    self.Q2next = self.fullQNet(self.nextstate, reuse=True)
                argmaxQnext = tf.cast(tf.argmax(self.Qnext, 1),dtype=tf.int32)
                argmaxQ2next = tf.cast(tf.argmax(self.Q2next, 1),dtype=tf.int32)
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
            if self.config['clip']>0 and "cliptype" in self.config and self.config['cliptype']=='deltaclip':
                self.errorlist = clipdelta(delta,self.config['clip'])
                print(self.errorlist.get_shape())
            else:
                pass#self.errorlist = 0.5 * delta ** 2  # tf.minimum(0.5 * delta ** 2, tf.abs(delta))  # (self.singleQ - self.y) ** 2
            #print('errorlist', self.errorlist.get_shape())
            if self.config['doubleQ']:
                if self.config['clip']>0 and "cliptype" in self.config and self.config['cliptype'] == 'deltaclip':
                    self.errorlist2 = clipdelta(delta2,self.config['clip'])
                else:
                    self.errorlist2 = 0.5 * delta2 ** 2
            #self.cost = tf.reduce_mean(self.errorlist)

            #regul=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,tf.get_variable_scope().name)
            #printRegularization()
            #if regul!=[]:
            #    self.cost+= tf.add_n(regul)
            #tf.summary.histogram('cost', self.cost)
            if self.config['doubleQ']:
                if regul != []:
                    self.logger.warning("doubleQ with regularization not properly implemented")
                self.cost2 = tf.reduce_mean(self.errorlist2)
                self.optimizer2 = tf.train.RMSPropOptimizer(self.learnrate, decay=self.config['decayoptimizer'],
                                momentum=self.config['momentum'], epsilon=self.config['epsoptimizer'])
                if self.config["clip"]>0 and self.config['cliptype'] != 'deltaclip':
                    self.optimizer2 = self.clip(self.optimizer2,self.cost2, global_step=self.global_step,lim=self.config['clip'])
                else:
                    self.optimizer2 = self.optimizer2.minimize(self.cost2, global_step=self.global_step)

            model_agent = self.Q
            self.optimizer = torch.optim.RMSprop(model_agent.parameters(), lr=self.learnrate,
                                        momentum=self.config['momentum'],
                                        weight_decay=self.config['regularization'][0],
                                        alpha=self.config['decayoptimizer'],
                                        eps=self.config['epsoptimizer'])

        self.coord = tf.train.Coordinator()
        self.workers = []

        self.numupdates = 0
        self.memoryLock = threading.Lock()

        self.memoryLock.acquire()
        self.memory = Memory(self.config['memsize'])
        self.memoryLock.release()

        print( (self.config['memsize'],)+tuple(n_input))

        #todo model loading
        # if self.config['file'] is None or (not os.path.isfile(self.config['file'] + ".p")):
        #     self.sess.run(tf.global_variables_initializer())
        # else:
        #     print("loading " + self.config['file'] + ".tf")
        #     self.saver.restore(self.sess, os.path.abspath(self.config['file'] + ".tf"))

        #self.sess.run(tf.assign(self.global_step, 0))


    # def evalQ(self, state, action):
    #     if self.isdiscrete:
    #         if state.ndim == 1:
    #             state = state.reshape(1, -1)
    #         return self.sess.run(self.singleQ, feed_dict={self.x: state, self.curraction: action})
    #     else:
    #         if state.ndim == 1:
    #             stateaction = np.concatenate((state.reshape(1, -1), action.reshape(1, -1)), 1)
    #         else:
    #             stateaction = np.concatenate((state, action), 1)
    #         return self.sess.run(self.Q, feed_dict={self.x: stateaction})

    def learn(self,force=False,print_iteration=False):
        self.Q.train()
        self.Q = self.Q.to(self.device_eval)
        update = (np.random.random() < self.config['probupdate'])
        if update or force:
            self.memoryLock.acquire()
            if self.memory.sizemem() > self.config['randstart']:
                if print_iteration and self.numupdates%100==0:
                    print('learn',self.numupdates)
                self.numupdates+=1
                ind = np.random.choice(self.memory.sizemem()-1, self.config['batch_size'])

                allstate=np.zeros((ind.shape[0],)+self.memory[ind[0]][0].shape)
                nextstates = np.zeros((ind.shape[0],) + self.memory[ind[0]][0].shape)
                currew = np.zeros((ind.shape[0],1))
                maxsteps_reward = np.zeros((ind.shape[0], 1))
                notdonevec = np.zeros((ind.shape[0], 1))
                allactionsparse = np.zeros((ind.shape[0], self.n_out))
                i=0
                #  [state, action, reward, notdone]

                for j in ind:
                    allstate[i]=self.memory[j][0]
                    if self.memory[j][3]==1:
                        nextstates[i]=self.memory[j+1][0]
                        #assert (nextstates[i]==self.memory[j][-1]).all()
                    currew[i,0]=self.memory[j][2]
                    notdonevec[i,0]=self.memory[j][3]
                    allactionsparse[i] = onehot(self.memory[j][1], self.n_out)

                    if self.config['lambda'] > 0 and (np.random.random()<0.01 or self.memory[j][6] is None):
                        limitd = 500
                        gamma = 1.
                        offset = 0
                        nextstate = 1 # next state relative index
                        # assuming nextstate available
                        #if nextstate == None:
                        #    alternativetarget += gamma * self.maxq(onew) * d
                        n = j
                        while nextstate != None and offset < limitd:
                            maxsteps_reward[i,0] += gamma * self.memory[n][2]
                            gamma = gamma * self.config['discount']
                            if n+nextstate*2>=self.memory.sizemem() or not (offset+nextstate < limitd) or self.memory[n][3]==0:
                                maxsteps_reward[i, 0] += gamma * self.maxq(self.memory[n+nextstate][0]) * self.memory[n][3]
                                nextstate = None
                            else:
                                offset += nextstate
                                n = j + offset
                        self.memory[j][6] = maxsteps_reward[i, 0]
                    else:
                        maxsteps_reward[i, 0] = self.memory[j][6]
                    i += 1

                self.memoryLock.release()
                flag = np.random.random() < 0.5 and self.fulldouble
                self.optimizer.zero_grad()
                self.Q.to(self.device)
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
                            self.y: alltarget.reshape( (-1, 1)),
                            self.curraction: allactionsparse})
                else:
                    allactionsparse = Variable(torch.from_numpy(allactionsparse).float()).to(self.device)
                    allstate = Variable(torch.from_numpy(allstate).float()).to(self.device)
                    nextstates = Variable(torch.from_numpy(nextstates).float()).to(self.device)
                    currew = Variable(torch.from_numpy(currew.reshape((-1,))).float()).to(self.device)
                    notdonevec = Variable(torch.from_numpy(notdonevec.reshape((-1,))).float()).to(self.device)
                    maxsteps_reward = Variable(torch.from_numpy(maxsteps_reward.reshape((-1,))).float()).to(self.device)
                    currQ = self.Q(allstate)
                    singleQ = torch.sum(allactionsparse * currQ, dim=1)
                    maxQnext = torch.max(self.Q(nextstates),dim=1)[0]
                    maxQnext = maxQnext.detach() #fixme
                    if self.config['episodic']:
                        target = (currew + self.config["discount"] * maxQnext * notdonevec) * (1. - self.config['lambda'])
                        #loss = self.criterion(singleQ,(currew + self.config["discount"] * maxQnext * notdonevec))
                    else:
                        target = (currew + self.config["discount"] * maxQnext)* (1. - self.config['lambda'])
                        #loss = self.criterion(singleQ, (currew + self.config["discount"] * maxQnext))
                    if self.config['lambda']>0:
                        target += maxsteps_reward * self.config['lambda']
                    loss = self.criterion(singleQ,target)
                    if self.config['exploss']>0:
                        loss += torch.mean(self.config['exploss'] * currQ)

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
                self.memoryLock.release()
        self.Q.to(self.device_eval)
        return self.numupdates



    def learnpolicy(self,startind=None,endind=None):
        self.memoryLock.acquire()
        # todo fix indexing with new indexes
        if startind is None or endind is None:
            startind = max(0, self.sizemem - self.config['memsize'])
            endind = self.sizemem
        n = endind - startind
        if n>=1: #self.sizemem >= self.config['batch_size']:
            allstate=np.zeros((n,)+self.memory[startind][0].shape)
            listdiscounts= np.zeros((n,1))

            nextstates = np.zeros((n,) + self.memory[startind][0].shape)
            currew = np.zeros((n,1))
            notdonevec = np.zeros((n, 1))
            allactionsparse = np.zeros((n, self.n_out))
            i=0
            #  [state, action, reward, notdone, t,w]
            Gtv = np.zeros((n, 1))
            if self.memory[endind-1][3]==1:
                totalRv = self.sess.run(self.V, feed_dict={
                    self.x: self.memory[endind - 1][0].reshape((1,) + self.memory[endind - 1][0].shape)})[0, 0]
                Gtv[0] = totalRv
            else:
                totalRv = self.memory[endind-1][2]
                Gtv[0] = self.memory[endind-1][2]
            weights = np.zeros((n,1))
            for j in range(endind-1,startind-1,-1):
                weights[i,0]=self.memory[j][5]
                if i>0:
                    totalRv*=self.config['discount']
                    totalRv += self.memory[j][2]
                    Gtv[i] = totalRv

                listdiscounts[i]=self.config['discount']**self.memory[j][4]
                allstate[i]=self.memory[j][0]
                if self.memory[j][3]==1 and j+1<self.sizemem:
                    nextstates[i]=self.memory[j+1][0]

                currew[i,0]=self.memory[j][2]
                notdonevec[i,0]=self.memory[j][3]
                allactionsparse[i] = onehot(self.memory[j][1], self.n_out)
                i += 1
            assert i==n
            Vallstate=self.sess.run(self.V, feed_dict={self.x: allstate})
            Vnext = np.append([[0]],Vallstate[:-1],axis=0)
            if self.config['episodic']:
                targetV = currew + self.config['discount'] * Vnext * notdonevec

                #targetQ = currew + self.config['discount'] * self.maxqbatch(nextstates).reshape(-1, 1) * notdonevec


                if notdonevec[0,0]==1:
                    targetV[0]=Vallstate[0]
                    targetV = targetV[1:]
                    Vallstate = Vallstate[1:]
                    Vnext = Vnext[1:]
                    listdiscounts = listdiscounts[1:]
                    Gtv = Gtv[1:]
                    allstate = allstate[1:]
                    allactionsparse = allactionsparse[1:]
                    weights = weights[1:]

                if False:
                    print(notdonevec[0,0])
                    print('currew', (currew).reshape(-1, )[0:5], (currew).reshape(-1, )[-5:])
                    print('Gtv', (Gtv).reshape(-1, )[0:5], (Gtv).reshape(-1, )[-5:])
                    print('targetV',(targetV).reshape(-1, )[0:5], (targetV).reshape(-1, )[-5:])
                    print('Vnext',self.sess.run(self.V, feed_dict={self.x: nextstates}).reshape(-1, )[0:5])
                    print('Vnext', Vnext.reshape(-1, )[0:5])
                    print('Vstate', Vallstate.reshape(-1, )[0:5])
                    print((notdonevec).reshape(-1, )[0:5], (notdonevec).reshape(-1, )[-5:])

                if np.random.random()<0.05:
                    print('Vstate', Vallstate.reshape(-1, )[0:5])
                targetV = targetV*(1-self.config['lambda']) + Gtv*self.config['lambda']
                #targetQ = targetQ*(1-self.config['lambda']) + Gt*self.config['lambda']
                if self.config['discounted_policy_grad']==False:
                    listdiscounts=1.
                if self.newpolicy:
                    QQ=self.sess.run(self.singleQ, feed_dict={self.x: allstate,self.curraction:allactionsparse}).reshape(-1, 1)
                    targetp = weights*listdiscounts * (QQ - Vallstate) #self.maxqbatch(allstate).reshape(-1, 1))
                else:
                    targetp = weights * listdiscounts * (targetV - Vallstate)
                #targetp = listdiscounts*(Gt- self.sess.run(self.V, feed_dict={self.x: allstate}))
            else:
                raise Exception('non-episodic not implemented')
                exit(-1)
            self.memoryLock.release()

            '''
            self.sess.run(self.optimizer, feed_dict={
                self.x: allstate,
                self.y: targetQ.reshape((-1, 1)),
                self.curraction: allactionsparse})
            '''

            self.sess.run(self.optimizer_policy, feed_dict={
                self.x: allstate,
                self.yp: targetp.reshape((-1, )),
                self.y: targetV.reshape((-1, 1)),
                self.curraction: allactionsparse})
            '''
            self.sess.run(self.optimizer_V, feed_dict={
                self.x: allstate,
                self.y: targetV.reshape((-1, 1))})
            '''
            return True
        else:
            self.memoryLock.release()
            return False


    def getweights(self, lb):
        return self.sess.run(tf.get_variable(lb))
    def getweights2(self, lb):
        return tf.get_variable(lb)

    def maxq(self, observation):
        self.Q.eval()
        if self.isdiscrete:
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            else:
                observation = observation.reshape(tuple([1] + list(observation.shape)))
            if self.copyQalone:
                return np.max(self.sess.run(self.Q2, feed_dict={self.x: observation})).reshape(1, )
            else:
                var_obs = Variable(torch.from_numpy(observation).float()).to(self.device_eval)
                currQ = self.Q(var_obs).cpu().data.numpy()
                return np.max(currQ).reshape(1, )
    def maxqbatch(self, observation):
        if self.isdiscrete:
            if self.copyQalone:
                return np.max(self.sess.run(self.Q2, feed_dict={self.x: observation}),1)
            else:
                return np.max(self.sess.run(self.Q, feed_dict={self.x: observation}),1)

    def copyq(self):

        self.sess.run(self.copyQ2_Q)
        '''
        for w in self.weights:
            self.sess.run(tf.assign(self.weights2[w], self.weights[w]))
        for b in self.biases:
            self.sess.run(tf.assign(self.biases2[b], self.biases[b]))
        '''
    def doublemaxqbatch(self, observation,flag):
        # print observation
        if self.isdiscrete:
            if flag:
                Q = self.sess.run(self.Q, feed_dict={self.x: observation})
                #print Q.shape,np.argmax(self.sess.run(self.Q2, feed_dict={self.x: observation}),1).shape,Q[np.argmax(self.sess.run(self.Q2, feed_dict={self.x: observation}),1)].shape
                return Q[range(Q.shape[0]),np.argmax(self.sess.run(self.Q2, feed_dict={self.x: observation}),1)]
            else:
                Q2 = self.sess.run(self.Q2, feed_dict={self.x: observation})
                #print Q2.shape,np.argmax(self.sess.run(self.Q, feed_dict={self.x: observation}),1).shape, Q2[np.argmax(self.sess.run(self.Q, feed_dict={self.x: observation}),1)].shape
                return Q2[range(Q2.shape[0]),np.argmax(self.sess.run(self.Q, feed_dict={self.x: observation}),1)]
    def argmaxq(self, observation):
        self.Q.eval()
        if self.isdiscrete:
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            else:
                observation = observation.reshape(tuple([1] + list(observation.shape)))
            # print observation,self.sess.run(self.Q, feed_dict={self.x:observation})
            if self.config['doubleQ'] and (self.fulldouble and np.random.random()<0.5):
                return np.argmax(self.sess.run(self.Q2, feed_dict={self.x: observation}))
            else:
                var_obs = Variable(torch.from_numpy(observation).float()).to(self.device_eval)
                currQ = self.Q(var_obs).cpu().data.numpy()
                return np.argmax(currQ)
                #return np.argmax(self.sess.run(self.Q, feed_dict={self.x: observation}))
    def evalQ(self, observation):
        self.Q.eval()
        if self.isdiscrete:
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            #else:
            #    observation = observation.reshape(tuple([1] + list(observation.shape)))
        var_obs = Variable(torch.from_numpy(observation).float()).to(self.device_eval)
        currQ = self.Q(var_obs).cpu().data.numpy()
        return currQ
    def softmaxq(self, observation):
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

    def act(self, observation, episode=None):
        self.Q.to(self.device_eval)
        eps = self.epsilon(episode)

        # epsilon greedy.
        if np.random.random() > eps:
            action = self.argmaxq(observation)  # self.softmaxq(observation)#
        # print self.softmaxq(observation)
        # print 'greedy',action
        else:
            action = self.action_space.sample()
        # print 'sample',action
        return action

    def actpolicy(self, observation, episode=None):
        prob=self.sess.run(self.policy, feed_dict={self.x: observation.reshape(tuple([1] + list(observation.shape)))})
        proposal = prob[0]#(prob[0]+0.03)/np.sum(prob[0]+0.03)
        if False and (episode is None or episode == -1):
            action = np.argmax(prob[0]) # disabled!
        else:
            action = np.random.choice(self.n_out, p=proposal)#(prob[0]+0.04)/np.sum(prob[0]+0.04))
        if np.random.random()<0.01:
            logit = self.sess.run(self.logitpolicy,
                                 feed_dict={self.x: observation.reshape(tuple([1] + list(observation.shape)))})
            print('prob',prob[0],logit[0])
        w = np.minimum(10.,prob[0][action]/proposal[action])
        return action,w

    def close(self):
        self.sess.close()
        tf.reset_default_graph()
