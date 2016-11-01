'''
Created on Jun 26, 2016

@author: Davide Nitti
'''

import gym
import logging
import numpy as np
import math, time
import tensorflow as tf
import pickle
from gym import spaces
import os.path


def multilayer_perceptron(_X, numhidden, regulariz, minout=None, maxout=None, initbias=0., outhidden=-1, seed=None):
    if seed is not None:
        tf.set_random_seed(seed)
    numlayers = len(numhidden)
    layer_i = _X
    regul = 0.
    for i in xrange(1, numlayers - 1):
        w = tf.Variable(
            tf.truncated_normal([numhidden[i - 1], numhidden[i]], stddev=1. / math.sqrt(float(numhidden[i - 1]))),
            name="w" + str(i))
        b = tf.Variable(tf.zeros([numhidden[i]]), name="b" + str(i))
        layer_i = tf.nn.tanh(tf.add(tf.matmul(layer_i, w), b))
        regul += tf.nn.l2_loss(w) * regulariz[i - 1]  # + tf.nn.l2_loss(b)*initbias
        if outhidden == i:
            hidlayer = layer_i
        print 'w', w.get_shape(), 'b', b.get_shape(), 'l', layer_i.get_shape(), regulariz[i - 1]
    w = tf.Variable(tf.truncated_normal([numhidden[numlayers - 2], numhidden[numlayers - 1]],
                                        stddev=1. / math.sqrt(float(numhidden[numlayers - 2]))),
                    name="w" + str(numlayers - 1))
    b = tf.Variable(tf.zeros([numhidden[numlayers - 1]]), name="b" + str(numlayers - 1))

    if minout == None:
        layer_out = tf.matmul(layer_i, w) + b + initbias
    # layer_out/=100.
    else:
        layer_out = tf.nn.sigmoid(tf.matmul(layer_i, w) + b + initbias) * (maxout - minout) + minout
    print regulariz, numlayers, len(numhidden)
    print 'w', w.get_shape(), 'b', b.get_shape(), 'l', layer_out.get_shape(), regulariz[numlayers - 2]
    regul += tf.nn.l2_loss(w) * regulariz[numlayers - 2]  # +tf.nn.l2_loss(b)*initbias
    if outhidden >= 0:
        return layer_out, regul, hidlayer
    else:
        return layer_out, regul


def onehot(i, n):
    out = np.zeros(n)
    out[i] = 1.
    return out


class deepQAgent(object):
    def __del__(self):
        print 'deepQAgent died'
        self.close()

    def save(self, filename=None):
        if filename is None:
            filename = self.config['file']
        with open(filename + ".p", "wb") as input_file:
            pickle.dump((self.observation_space, self.action_space, self.reward_range, self.config), input_file)
        save_path = self.saver.save(self.sess, filename + ".tf")

    def __init__(self, observation_space, action_space, reward_range, **userconfig):
        if userconfig["file"] is not None and os.path.isfile(userconfig["file"] + ".p"):
            with open(userconfig["file"] + ".p", "rb") as input_file:
                self.observation_space, self.action_space, self.reward_range, self.config = pickle.load(input_file)
                # overwrite some parameter
                self.config["initial_learnrate"] = userconfig["initial_learnrate"]
                self.config["eps"] = userconfig["eps"]
                self.config["batch_size"] = userconfig["batch_size"]
                self.config["probupdate"] = userconfig["probupdate"]
                self.config["lambda"] = userconfig["lambda"]
                self.config["momentum"] = userconfig["momentum"]
                self.config["memsize"] = userconfig["memsize"]
                if not ("featureset" in self.config):
                    self.config["featureset"] = userconfig["featureset"]
                    # self.config["fee"]=userconfig["fee"]
                    # self.config["minfee"]=userconfig["minfee"]
                    # self.config["maxfee"]=userconfig["maxfee"]
                    # self.config["multiplebuy"]=userconfig["multiplebuy"]
                    # self.config["numstock"]=userconfig["numstock"]
        else:
            self.observation_space = observation_space
            self.action_space = action_space
            self.reward_range = reward_range
            self.config = {
                "memsize": 50000,
                "scalereward": 1.,
                "probupdate": 0.25,
                "lambda": 0.1,
                "past": 0,
                "eps": 0.15,  # Epsilon in epsilon greedy policies
                "decay": 0.996,  # Epsilon decay in epsilon greedy policies
                "initial_learnrate": 0.008,
                "decay_learnrate": 0.999,
                "discount": 0.99,
                "batch_size": 75,
                "hiddenlayers": [300],
                "regularization": [0.0001, 0.000001],
                "momentum": 0.1,
                "file": None,
                "seed": None}
            self.config.update(userconfig)

        if self.config["seed"] is not None:
            np.random.seed(self.config["seed"])
            print "seed", self.config["seed"]
        # print self.config["initial_learnrate"]
        self.isdiscrete = isinstance(self.action_space, gym.spaces.Discrete)
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise Exception('Observation space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(
                observation_space, self))

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.learnrate = tf.train.exponential_decay(self.config['initial_learnrate'], self.global_step, 100,
                                                    self.config['decay_learnrate'], staircase=True, name="learnrate")
        # self.config['learning_rate']=tf.Variable(self.config['initial_learnrate'],trainable=False)
        self.bias = tf.Variable(0.0, trainable=False, name="bias")

        self.initQnetwork()

    def scaleobs(self, obs):
        if np.isinf(self.observation_space.low).any() or np.isinf(self.observation_space.high).any():
            return obs
        else:  # from -3 to 3
            o = (obs - self.observation_space.low) / (
                self.observation_space.high - self.observation_space.low) * 2. - 1.
            return o * 3

    def epsilon(self, episode=None):
        if episode == None:
            return 0.
        else:
            return self.config['eps'] * self.config['decay'] ** episode
            #	def setlearnrate(self,l):
            #		self.sess.run(tf.assign(self.config['learning_rate'],self.config['learning_rate']*l))

    def incbias(self, l):
        self.sess.run(tf.assign(self.bias, self.bias + l))

    def getbias(self):
        return self.sess.run(self.bias)

    def getlearnrate(self):
        return self.sess.run(self.learnrate)

    def stateautoencoder(self, n_input):
        self.plotautoenc = False
        self.outstate, regul, self.hiddencode = multilayer_perceptron(self.sa, [n_input, 2, n_input],
                                                                      [0.000001, 0.000001], outhidden=1,
                                                                      seed=self.config["seed"])
        # print self.x.get_shape(),self.outstate.get_shape()
        self.costautoenc = tf.reduce_mean((self.sa - self.outstate) ** 2) + regul
        self.optimizerautoenc = tf.train.RMSPropOptimizer(self.learnrate, 0.9, 0.05).minimize(self.costautoenc,
                                                                                              global_step=self.global_step)

    def initQnetwork(self):
        if self.isdiscrete:
            n_input = self.observation_space.shape[0] * (self.config['past'] + 1)
            self.n_out = self.action_space.n

        self.x = tf.placeholder("float", [None, n_input], name="self.x")
        self.y = tf.placeholder("float", [None, 1], name="self.y")
        self.yR = tf.placeholder("float", [None, 1], name="self.yR")

        print 'obs', n_input, 'action', self.n_out
        self.Qrange = (self.reward_range[0] * 1. / (1. - self.config['discount']),
                       self.reward_range[1] * 1. / (1. - self.config['discount']))
        print self.Qrange
        # self.scale=200./max(abs(self.Qrange[1]),abs(self.Qrange[0]))
        self.Q, regul = multilayer_perceptron(self.x, [n_input] + self.config['hiddenlayers'] + [self.n_out],
                                              self.config['regularization'],
                                              initbias=.0, seed=self.config["seed"])  # ,self.Qrange[0],self.Qrange[1])

        self.R, regulR = multilayer_perceptron(self.x, [n_input, 100, self.n_out], self.config['regularization'],
                                               initbias=.0, seed=self.config["seed"])  # ,self.Qrange[0],self.Qrange[1])

        if self.isdiscrete:
            self.curraction = tf.placeholder("float", [None, self.n_out], name="curraction")
            # index = tf.concat(0, [self.out, tf.constant([0, 0, 0], tf.int64)])
            self.singleQ = tf.reduce_sum(self.curraction * self.Q,
                                         reduction_indices=1)  # tf.reduce_sum(input_tensor, reduction_indices, keep_dims, name)#tf.slice(self.Q, [0,self.curraction],[-1,1])  #self.Q[:,self.out]
            self.singleQ = tf.reshape(self.singleQ, [-1, 1])

            self.singleR = tf.reduce_sum(self.curraction * self.R,
                                         reduction_indices=1)  # tf.reduce_sum(input_tensor, reduction_indices, keep_dims, name)#tf.slice(self.Q, [0,self.curraction],[-1,1])  #self.Q[:,self.out]
            self.singleR = tf.reshape(self.singleR, [-1, 1])

            # print 'singleR',self.singleR.get_shape()
            self.errorlistR = (self.singleR - self.yR) ** 2
            self.errorlist = (self.singleQ - self.y) ** 2

        self.cost = tf.reduce_mean(self.errorlist) + regul
        self.costR = tf.reduce_mean(self.errorlistR) + regulR
        self.lastcost = 0.
        self.optimizer = tf.train.RMSPropOptimizer(self.learnrate, 0.9, self.config['momentum']).minimize(self.cost,
                                                                                                          global_step=self.global_step)

        self.optimizerR = tf.train.RMSPropOptimizer(self.learnrate, 0.9, 0.).minimize(self.costR,
                                                                                      global_step=self.global_step)

        self.sa = tf.placeholder("float", [None, n_input + self.n_out], name="sa")
        self.stateautoencoder(n_input + self.n_out)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.memory = []
        self.errmemory = []
        if self.config['file'] is None or (not os.path.isfile(self.config['file'] + ".tf")):
            self.sess.run(tf.initialize_all_variables())
        else:
            print "loading " + self.config['file'] + ".tf"
            self.saver.restore(self.sess, self.config['file'] + ".tf")
        self.sess.run(tf.assign(self.global_step, 0))

    def evalQ(self, state, action):
        if self.isdiscrete:
            if state.ndim == 1:
                state = state.reshape(1, -1)
            return self.sess.run(self.singleQ, feed_dict={self.x: state, self.curraction: action})
        else:
            if state.ndim == 1:
                stateaction = np.concatenate((state.reshape(1, -1), action.reshape(1, -1)), 1)
            else:
                stateaction = np.concatenate((state, action), 1)
            return self.sess.run(self.Q, feed_dict={self.x: stateaction})

    def learnR(self, state, action, obnew, reward, notdone, nextaction):
        if self.isdiscrete:
            allstate = state.reshape(1, -1)
            allaction = np.array([action])
            alltarget = np.array([obnew]).reshape(1, -1)
            alldone = [notdone]
            allR = np.array([reward])
            indexes = [-1]
            update = (np.random.random() < .04)
            # cost1=self.sess.run(self.cost, feed_dict={self.x:allstateaction,self.y:alltarget})/allstateaction.shape[0]
            # erist=self.sess.run(self.errorlist, feed_dict={self.x:allstateaction,self.y:alltarget})
            # print erist
            if update:
                if len(self.memory) > self.config['batch_size']:
                    ind = np.random.choice(len(self.memory), self.config['batch_size'],
                                           replace=False)  # ,p=np.array(self.errmemory)/sum(self.errmemory))
                    # ind2=np.random.choice(len(self.memory), self.config['batch_size']/2, replace=False,p=np.array(self.errmemory)/sum(self.errmemory))
                    # ind=np.concatenate((ind,ind2))
                    for j in ind:
                        s, a, r, onew, d, _, nextstate = self.memory[j]
                        alltarget = np.concatenate((alltarget, obnew.reshape(1, -1)),
                                                   0)  # =np.concatenate((alltarget,self.config['discount']*self.maxqR(onew)*d ),0) #np.concatenate((alltarget,r+self.config['discount']*self.maxq(onew)*d ),0)
                        alldone.append(d)
                        allR = np.concatenate((allR, np.array([r])), 0)
                        # if np.random.random()<1.:
                        #	self.memory[j]=(s,a,r,onew,d,r+self.config['discount']*self.maxq(onew).reshape(1,)*d )
                        allstate = np.concatenate((allstate, s.reshape(1, -1)), 0)
                        allaction = np.concatenate((allaction, np.array([a])), 0)
                        indexes.append(j)
                    allactionsparse = np.zeros((allstate.shape[0], self.n_out))
                    allactionsparse[np.arange(allaction.shape[0]), allaction] = 1.
                    # print np.array(alldone).shape,self.maxqR(alltarget).shape
                    alltarget = self.config['discount'] * self.maxqR(alltarget) * np.array(alldone)
                    #	print alltarget[0:2]
                    self.sess.run(self.optimizer, feed_dict={self.x: allstate, self.y: alltarget.reshape((-1, 1)),
                                                             self.curraction: allactionsparse})
                    self.sess.run(self.optimizerautoenc, feed_dict={self.sa: allstate, self.outstate: allstate})
                    self.sess.run(self.optimizerR, feed_dict={self.x: allstate, self.yR: allR.reshape((-1, 1)),
                                                              self.curraction: allactionsparse})

                    if False:
                        c = self.sess.run(self.cost, feed_dict={self.x: allstate, self.y: alltarget.reshape((-1, 1)),
                                                                self.curraction: allactionsparse})
                        if c > self.lastcost:
                            self.setlearnrate(0.999)
                        else:
                            self.setlearnrate(1.001)
                        self.lastcost = c
                else:
                    alltarget = self.config['discount'] * self.maxqR(alltarget) * np.array(alldone)
                    allactionsparse = np.zeros((allstate.shape[0], self.n_out))
                    allactionsparse[np.arange(allaction.shape[0]), allaction] = 1.
            else:
                alltarget = self.config['discount'] * self.maxqR(alltarget) * np.array(alldone)
                allactionsparse = np.zeros((allstate.shape[0], self.n_out))
                allactionsparse[np.arange(allaction.shape[0]), allaction] = 1.
            indexes = np.array(indexes)

            erlist = self.sess.run(self.errorlist, feed_dict={self.x: allstate, self.y: alltarget.reshape((-1, 1)),
                                                              self.curraction: allactionsparse})
            erlist = erlist.reshape(-1, )
            self.errmemory.append(erlist[0])

            for i, er in enumerate(erlist):
                # print indexes[i], er,self.errmemory[indexes[i]]
                self.errmemory[indexes[i]] = er  # self.errmemory[a]
            self.errmemory = self.errmemory[-50000:]

            if len(self.memory) > 0 and np.array_equal(self.memory[-1][3], state):
                self.memory[-1][6] = len(self.memory)
            self.memory.append([state, action, reward, obnew, notdone, 0, None])

            self.memory = self.memory[-50000:]
            # print self.sess.run(self.cost, feed_dict={self.x:allstateaction,self.y:alltarget})-cost1
            return 0  # self.sess.run(self.costautoenc,feed_dict={self.x:allstate,self.outstate:allstate}) #self.sess.run(self.cost, feed_dict={self.x:allstate,self.y:alltarget.reshape((-1,1)),self.curraction:allactionsparse})/allstate.shape[0]

    def learn(self, state, action, obnew, reward, notdone, nextaction):
        if self.isdiscrete:
            target = reward + self.config['discount'] * self.maxq(obnew) * notdone
            # target=reward+self.config['discount']*self.evalQ(obnew,nextaction)*notdone

            target = target.reshape(1, )
            allstate = state.reshape(1, -1)
            allaction = np.array([action])
            alltarget = target
            # allR=np.array([reward])
            indexes = [-1]
            update = (np.random.random() < self.config['probupdate'])
            # cost1=self.sess.run(self.cost, feed_dict={self.x:allstateaction,self.y:alltarget})/allstateaction.shape[0]
            # erist=self.sess.run(self.errorlist, feed_dict={self.x:allstateaction,self.y:alltarget})
            # print erist
            if update:
                if len(self.memory) > self.config['batch_size']:
                    # p=np.arange(4,step=4./len(self.memory))+1.
                    # p=p[0:len(self.memory)]
                    # p/=np.sum(p)
                    ind = np.random.choice(len(self.memory), self.config[
                        'batch_size'])  # ,p=np.array(self.errmemory)/sum(self.errmemory))
                    # ind2=np.random.choice(len(self.memory), self.config['batch_size']/2, replace=False,p=np.array(self.errmemory)/sum(self.errmemory))
                    # ind=np.concatenate((ind,ind2))
                    for j in ind:
                        s, a, r, onew, d, Q, nextstate = self.memory[j]

                        if Q != None:
                            alternativetarget = Q
                        else:
                            if self.config['lambda'] > 0.:
                                limitd = 1000
                                alternativetarget = r
                                gamma = self.config['discount']
                                offset = 0
                                if nextstate == None:
                                    alternativetarget += gamma * self.maxq(onew) * d
                                # besttarget=(r+gamma*self.maxq(onew) * d)[0]

                                while nextstate != None and offset < limitd:
                                    offset += nextstate
                                    n = j + offset

                                    # print j,n
                                    alternativetarget += gamma * self.memory[n][2]
                                    gamma = gamma * self.config['discount']
                                    # print j,n,self.memory[n][6]
                                    # besttarget=max(besttarget,(alternativetarget+gamma*self.maxq(self.memory[n][3]) * self.memory[n][4])[0])
                                    if self.memory[n][6] == None or not (offset < limitd):
                                        alternativetarget += gamma * self.maxq(self.memory[n][3]) * self.memory[n][4]
                                    # if self.memory[n][4]<0.5:
                                    #	print alternativetarget,self.memory[n][4]
                                    nextstate = self.memory[n][6]
                                    # print besttarget.shape,np.max(besttarget)
                            else:
                                alternativetarget = 0.
                            self.memory[j][5] = alternativetarget
                        # print besttarget

                        alternativetarget = alternativetarget * self.config['lambda'] + (r + self.config[
                            'discount'] * self.maxq(onew) * d) * (1. - self.config['lambda'])

                        alltarget = np.concatenate((alltarget, alternativetarget),
                                                   0)  # r+self.config['discount']*self.maxq(onew)*d   np.concatenate((alltarget,r+self.config['discount']*self.maxq(onew)*d ),0)
                        # allR=np.concatenate((allR,np.array([r])),0)
                        # if np.random.random()<1.:
                        #	self.memory[j]=(s,a,r,onew,d,r+self.config['discount']*self.maxq(onew).reshape(1,)*d )
                        allstate = np.concatenate((allstate, s.reshape(1, -1)), 0)
                        allaction = np.concatenate((allaction, np.array([a])), 0)
                        indexes.append(j)
                    allactionsparse = np.zeros((allstate.shape[0], self.n_out))
                    allactionsparse[np.arange(allaction.shape[0]), allaction] = 1.

                    self.sess.run(self.optimizer, feed_dict={self.x: allstate, self.y: alltarget.reshape((-1, 1)),
                                                             self.curraction: allactionsparse})

                    if self.plotautoenc:
                        sa = np.concatenate((allstate, allactionsparse), 1)
                        self.sess.run(self.optimizerautoenc, feed_dict={self.sa: sa, self.outstate: sa})
                    # self.sess.run(self.optimizerR, feed_dict={self.x:allstate,self.yR:allR.reshape((-1,1)),self.curraction:allactionsparse})
                    if False:
                        c = self.sess.run(self.cost, feed_dict={self.x: allstate, self.y: alltarget.reshape((-1, 1)),
                                                                self.curraction: allactionsparse})

                        if c > self.lastcost:
                            self.setlearnrate(0.9999)
                        else:
                            self.setlearnrate(1.0001)
                        self.lastcost = c
            allactionsparse = np.zeros((allstate.shape[0], self.n_out))
            allactionsparse[np.arange(allaction.shape[0]), allaction] = 1.
            indexes = np.array(indexes)
            erlist = self.sess.run(self.errorlist, feed_dict={self.x: allstate, self.y: alltarget.reshape((-1, 1)),
                                                              self.curraction: allactionsparse})
            erlist = erlist.reshape(-1, )
            self.errmemory.append(erlist[0])

            for i, er in enumerate(erlist):
                # print indexes[i], er,self.errmemory[indexes[i]]
                self.errmemory[indexes[i]] = er  # self.errmemory[a]
            self.errmemory = self.errmemory[-self.config['memsize']:]

            if len(self.memory) > 0 and np.array_equal(self.memory[-1][3], state):
                self.memory[-1][6] = 1
            self.memory.append([state, action, reward, obnew, notdone, None, None])
            '''
			if len(self.memory)>4:
				print len(self.memory)-4,self.memory[ self.memory[-4][6] ]
				print len(self.memory)-3,self.memory[-3]
				print len(self.memory)-2,self.memory[-2]
			'''
            self.memory = self.memory[-self.config['memsize']:]
            # print self.sess.run(self.cost, feed_dict={self.x:allstate,self.y:alltarget.reshape((-1,1)),self.curraction:allactionsparse})

            # print self.sess.run(self.cost, feed_dict={self.x:allstateaction,self.y:alltarget})-cost1
            return 0  # self.sess.run(self.costautoenc,feed_dict={self.x:allstate,self.outstate:allstate}) #self.sess.run(self.cost, feed_dict={self.x:allstate,self.y:alltarget.reshape((-1,1)),self.curraction:allactionsparse})/allstate.shape[0]

    def maxq(self, observation):
        if self.isdiscrete:
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            return np.max(self.sess.run(self.Q, feed_dict={self.x: observation})).reshape(1, )

    def argmaxq(self, observation):
        # print observation
        if self.isdiscrete:
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            # print observation,self.sess.run(self.Q, feed_dict={self.x:observation})
            return np.argmax(self.sess.run(self.Q, feed_dict={self.x: observation}))

    def softmaxq(self, observation):
        if self.isdiscrete:
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            G = self.sess.run(self.Q, feed_dict={self.x: observation})
            p = np.exp(G * 5) / np.sum(np.exp(G * 5))
            p = p.reshape(-1, )
            return np.random.choice(p.shape[0], 1, p=p)[0]
        else:
            print 'not implemented'
            exit(0)

    def maxqR(self, observation):
        if self.isdiscrete:
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            return np.max(self.sess.run(self.R, feed_dict={self.x: observation}) + self.sess.run(self.Q, feed_dict={
                self.x: observation}), 1).reshape(-1, )
        else:
            print 'not implemented'
            exit(0)

    def argmaxqR(self, observation):
        if self.isdiscrete:
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            return np.argmax(self.sess.run(self.R, feed_dict={self.x: observation}) + self.sess.run(self.Q, feed_dict={
                self.x: observation}))
        else:
            print 'not implemented'
            exit(0)

    def act(self, observation, episode=None):
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

    def actR(self, observation, episode):
        eps = self.epsilon(episode)
        # epsilon greedy.
        if np.random.random() > eps:
            action = self.argmaxqR(observation)
        # print 'greedy',action
        else:
            action = self.action_space.sample()
        # print 'sample',action
        return action

    def close(self):
        self.sess.close()


class deepQAgentCont(object):
    def __del__(self):
        print self.id, 'died'
        self.close()

    def __init__(self, observation_space, action_space, reward_range, **userconfig):
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_range = reward_range
        # self.isdiscrete = isinstance(self.action_space, gym.spaces.Discrete)
        if not isinstance(self.action_space, gym.spaces.Box):
            raise Exception(
                'Observation space {} incompatible with {}. (Only supports Continuous action spaces.)'.format(
                    observation_space, self))
        self.config = {
            "memsize": 50000,
            "scalereward": 1.,
            "probupdate": 0.25,
            "lambda": 0.1,
            "past": 0,
            "eps": 0.15,  # Epsilon in epsilon greedy policies
            "decay": 0.996,  # Epsilon decay in epsilon greedy policies
            "initial_learnrate": 0.008,
            "decay_learnrate": 0.999,
            "discount": 0.99,
            "batch_size": 75,
            "hiddenlayers": [300],
            "regularization": [0.001, 0.000001]}
        self.config.update(userconfig)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.learnrate = tf.train.exponential_decay(self.config['initial_learnrate'], self.global_step, 100,
                                                    self.config['decay_learnrate'], staircase=True, name="learnrate")
        # self.config['learning_rate']=tf.Variable(self.config['initial_learnrate'],trainable=False)
        self.bias = tf.Variable(0.0, trainable=False, name="bias")
        self.initQnetwork()

    def scaleobs(self, obs):
        if np.isinf(self.observation_space.low).any() or np.isinf(self.observation_space.high).any():
            return obs
        else:  # from -3 to 3
            o = (obs - self.observation_space.low) / (
                self.observation_space.high - self.observation_space.low) * 2. - 1.
            return o * 3

    def epsilon(self, episode=None):
        if episode == None:
            return 0.
        else:
            return self.config['eps'] * self.config['decay'] ** episode
            #	def setlearnrate(self,l):
            #		self.sess.run(tf.assign(self.config['learning_rate'],self.config['learning_rate']*l))

    def incbias(self, l):
        self.sess.run(tf.assign(self.bias, self.bias + l))

    def getbias(self):
        return self.sess.run(self.bias)

    def getlearnrate(self):
        return self.sess.run(self.learnrate, global_step=self.global_step)

    def stateautoencoder(self, n_input):
        self.plotautoenc = False
        self.outstate, regul, self.hiddencode = multilayer_perceptron(self.sa, [n_input, 2, n_input],
                                                                      [0.000001, 0.000001], outhidden=1,
                                                                      seed=self.config["seed"])
        # print self.x.get_shape(),self.outstate.get_shape()
        self.costautoenc = tf.reduce_mean((self.sa - self.outstate) ** 2) + regul
        self.optimizerautoenc = tf.train.RMSPropOptimizer(self.learnrate, 0.9, 0.05).minimize(self.costautoenc,
                                                                                              global_step=self.global_step)

    def initQnetwork(self):
        n_input = self.observation_space.shape[0] + self.action_space.shape[0]
        self.n_out = 1
        self.x = tf.placeholder("float", [None, n_input])
        self.y = tf.placeholder("float", [None, 1])
        self.yR = tf.placeholder("float", [None, 1])
        print 'obs', n_input, 'action', self.n_out
        self.Qrange = (self.reward_range[0] * 1. / (1. - self.config['discount']),
                       self.reward_range[1] * 1. / (1. - self.config['discount']))
        print self.Qrange
        # self.scale=200./max(abs(self.Qrange[1]),abs(self.Qrange[0]))
        self.Q, regul = multilayer_perceptron(self.x, [n_input] + self.config['hiddenlayers'] + [self.n_out],
                                              self.config['regularization'],
                                              initbias=.0, seed=self.config["seed"])  # ,self.Qrange[0],self.Qrange[1])

        # self.R,regulR = multilayer_perceptron(self.x,[n_input,100,self.n_out],penhidden,penout,initbias=.0)#,self.Qrange[0],self.Qrange[1])


        self.errorlist = (self.Q - self.y) ** 2
        self.cost = tf.reduce_mean(self.errorlist) + regul
        # self.costR = tf.reduce_mean(self.errorlistR)+regulR
        self.lastcost = 0.
        self.optimizer = tf.train.RMSPropOptimizer(self.learnrate, 0.9, 0.).minimize(self.cost,
                                                                                     global_step=self.global_step)

        # self.optimizerR = tf.train.RMSPropOptimizer(self.learnrate,0.9,0.).minimize(self.costR,global_step=self.global_step)

        self.sa = tf.placeholder("float", [None, n_input + self.n_out])
        # self.stateautoencoder(n_input+self.n_out)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.memory = []
        self.errmemory = []

    def evalQ(self, state, action):
        if state.ndim == 1:
            stateaction = np.concatenate((state.reshape(1, -1), action.reshape(1, -1)), 1)
        else:
            stateaction = np.concatenate((state, action), 1)
        return self.sess.run(self.Q, feed_dict={self.x: stateaction})

    def learn(self, state, action, obnew, reward, notdone, nextaction):
        if self.isdiscrete:
            target = reward + self.config['discount'] * self.maxq(obnew) * notdone
            # target=reward+self.config['discount']*self.evalQ(obnew,nextaction)*notdone

            target = target.reshape(1, )
            allstate = state.reshape(1, -1)
            allaction = np.array([action])
            alltarget = target
            # allR=np.array([reward])
            indexes = [-1]
            update = (np.random.random() < self.config['probupdate'])
            # cost1=self.sess.run(self.cost, feed_dict={self.x:allstateaction,self.y:alltarget})/allstateaction.shape[0]
            # erist=self.sess.run(self.errorlist, feed_dict={self.x:allstateaction,self.y:alltarget})
            # print erist
            if update:
                if len(self.memory) > self.config['batch_size']:
                    # p=np.arange(4,step=4./len(self.memory))+1.
                    # p=p[0:len(self.memory)]
                    # p/=np.sum(p)
                    ind = np.random.choice(len(self.memory), self.config[
                        'batch_size'])  # ,p=np.array(self.errmemory)/sum(self.errmemory))
                    # ind2=np.random.choice(len(self.memory), self.config['batch_size']/2, replace=False,p=np.array(self.errmemory)/sum(self.errmemory))
                    # ind=np.concatenate((ind,ind2))
                    for j in ind:
                        s, a, r, onew, d, Q, nextstate = self.memory[j]

                        if self.config['lambda'] > 0.:
                            limitd = 200
                            alternativetarget = r
                            gamma = self.config['discount']
                            offset = 0
                            if nextstate == None:
                                alternativetarget += gamma * self.maxq(onew) * d
                            while nextstate != None and offset < limitd:
                                offset += nextstate
                                n = j + offset

                                # print j,n
                                alternativetarget += gamma * self.memory[n][2]
                                gamma = gamma * self.config['discount']
                                # print j,n,self.memory[n][6]
                                if self.memory[n][6] == None or not (offset < limitd):
                                    alternativetarget += gamma * self.maxq(self.memory[n][3]) * self.memory[n][4]
                                # if self.memory[n][4]<0.5:
                                #	print alternativetarget,self.memory[n][4]
                                nextstate = self.memory[n][6]
                        else:
                            alternativetarget = 0.
                        alternativetarget = alternativetarget * self.config['lambda'] + (r + self.config[
                            'discount'] * self.maxq(onew) * d) * (1. - self.config['lambda'])

                        alltarget = np.concatenate((alltarget, alternativetarget),
                                                   0)  # r+self.config['discount']*self.maxq(onew)*d   np.concatenate((alltarget,r+self.config['discount']*self.maxq(onew)*d ),0)
                        # allR=np.concatenate((allR,np.array([r])),0)
                        # if np.random.random()<1.:
                        #	self.memory[j]=(s,a,r,onew,d,r+self.config['discount']*self.maxq(onew).reshape(1,)*d )
                        allstate = np.concatenate((allstate, s.reshape(1, -1)), 0)
                        allaction = np.concatenate((allaction, np.array([a])), 0)
                        indexes.append(j)
                    allactionsparse = np.zeros((allstate.shape[0], self.n_out))
                    allactionsparse[np.arange(allaction.shape[0]), allaction] = 1.

                    self.sess.run(self.optimizer, feed_dict={self.x: allstate, self.y: alltarget.reshape((-1, 1)),
                                                             self.curraction: allactionsparse})

                    if self.plotautoenc:
                        sa = np.concatenate((allstate, allactionsparse), 1)
                        self.sess.run(self.optimizerautoenc, feed_dict={self.sa: sa, self.outstate: sa})
                    # self.sess.run(self.optimizerR, feed_dict={self.x:allstate,self.yR:allR.reshape((-1,1)),self.curraction:allactionsparse})
                    if False:
                        c = self.sess.run(self.cost, feed_dict={self.x: allstate, self.y: alltarget.reshape((-1, 1)),
                                                                self.curraction: allactionsparse})

                        if c > self.lastcost:
                            self.setlearnrate(0.9999)
                        else:
                            self.setlearnrate(1.0001)
                        self.lastcost = c
            allactionsparse = np.zeros((allstate.shape[0], self.n_out))
            allactionsparse[np.arange(allaction.shape[0]), allaction] = 1.
            indexes = np.array(indexes)
            erlist = self.sess.run(self.errorlist, feed_dict={self.x: allstate, self.y: alltarget.reshape((-1, 1)),
                                                              self.curraction: allactionsparse})
            erlist = erlist.reshape(-1, )
            self.errmemory.append(erlist[0])

            for i, er in enumerate(erlist):
                # print indexes[i], er,self.errmemory[indexes[i]]
                self.errmemory[indexes[i]] = er  # self.errmemory[a]
            self.errmemory = self.errmemory[-self.config['memsize']:]

            if len(self.memory) > 0 and np.array_equal(self.memory[-1][3], state):
                self.memory[-1][6] = 1
            self.memory.append([state, action, reward, obnew, notdone, target, None])
            '''
			if len(self.memory)>4:
				print len(self.memory)-4,self.memory[ self.memory[-4][6] ]
				print len(self.memory)-3,self.memory[-3]
				print len(self.memory)-2,self.memory[-2]
			'''
            self.memory = self.memory[-self.config['memsize']:]
            # print self.sess.run(self.cost, feed_dict={self.x:allstate,self.y:alltarget.reshape((-1,1)),self.curraction:allactionsparse})

            # print self.sess.run(self.cost, feed_dict={self.x:allstateaction,self.y:alltarget})-cost1
            return 0  # self.sess.run(self.costautoenc,feed_dict={self.x:allstate,self.outstate:allstate}) #self.sess.run(self.cost, feed_dict={self.x:allstate,self.y:alltarget.reshape((-1,1)),self.curraction:allactionsparse})/allstate.shape[0]
        else:
            target = reward + self.config['discount'] * self.maxq(obnew) * notdone
            stateaction = np.concatenate((state.reshape(1, -1), action.reshape(1, -1)), 1)
            allstateaction = stateaction
            alltarget = target
            # cost1=self.sess.run(self.cost, feed_dict={self.x:allstateaction,self.y:alltarget})/allstateaction.shape[0]
            # erist=self.sess.run(self.errorlist, feed_dict={self.x:allstateaction,self.y:alltarget})
            # print erist
            if (np.random.random() < .5):
                if len(self.memory) > self.config['batch_size']:
                    ind = np.random.choice(len(self.memory), self.config['batch_size'], replace=False)
                    for j in ind:
                        sa, r, onew, d = self.memory[j]
                        alltarget = np.concatenate((alltarget, r + self.config['discount'] * self.maxq(onew) * d), 0)
                        allstateaction = np.concatenate((allstateaction, sa), 0)
                alltarget = alltarget.reshape((-1, 1))
                self.sess.run(self.optimizer, feed_dict={self.x: allstateaction, self.y: alltarget.reshape((-1, 1))})
            '''
			erlist=self.sess.run(self.errorlist, feed_dict={self.x:allstateaction,self.y:alltarget}).reshape(-1,)
			maxer=erlist.argsort()[-self.config['batch_size']:]
			if(np.random.random()<1.):
				for j in maxer:
					sa,r,onew,d=self.memory[j]
					alltarget=np.concatenate((alltarget,r+self.config['discount']*self.maxq(onew)*d ),0)
					allstateaction=np.concatenate((allstateaction,sa),0)
			'''
            # print erlist.shape,alltarget.shape,maxer.shape# erlist[maxer]
            # print erlist[maxer]
            # print self.sess.run(self.Q, feed_dict={self.x:allstateaction,self.y:alltarget}).shape
            # print 'all',alltarget.shape,allstateaction.shape
            self.memory.append((stateaction, reward, obnew, notdone))
            self.memory = self.memory[-100000:]

            # print self.sess.run(self.cost, feed_dict={self.x:allstateaction,self.y:alltarget})-cost1
            return 0  # self.sess.run(self.cost, feed_dict={self.x:allstateaction,self.y:alltarget.reshape((-1,1))})/allstateaction.shape[0]

    def maxq(self, observation):
        stateaction = np.array([np.concatenate((observation, self.action_space.sample())) for _ in xrange(50)])
        # print stateaction
        q = self.sess.run(self.Q, feed_dict={self.x: stateaction})
        # print stateaction[np.argmax(q)]
        # print q[np.argmax(q)] , stateaction[np.argmax(q),observation.shape[0]:]
        # print q.shape,observation.shape[0],np.argmax(q),stateaction[np.argmax(q),observation.shape[0]:]
        # print q[np.argmax(q)],np.max(q).reshape(1,)
        return np.max(q).reshape(1, )  # q[np.argmax(q)]

    def argmaxq(self, observation):
        # print observation
        stateaction = np.array([np.concatenate((observation, self.action_space.sample())) for _ in xrange(50)])
        q = self.sess.run(self.Q, feed_dict={self.x: stateaction})
        return stateaction[np.argmax(q), observation.shape[0]:]

    def act(self, observation, episode=None):
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

    def close(self):
        self.sess.close()


def do_rollout(agent, env, episode, num_steps=None, render=False):
    if num_steps == None:
        num_steps = env.spec.timestep_limit
    total_rew = 0.
    cost = 0.

    ob = env.reset()
    ob = agent.scaleobs(ob)
    ob1 = np.copy(ob)
    for _ in range(agent.config["past"]):
        ob1 = np.concatenate((ob1, ob))
    # print ob.shape,ob1.shape
    listob = [ob1]
    listact = []
    for t in range(num_steps):

        # start = time.time()
        a = agent.act(ob1, episode)
        # print 'time actR',time.time()-start
        (obnew, reward, done, _info) = env.step(a)
        obnew = agent.scaleobs(obnew)
        # print t,a,_info
        reward *= agent.config['scalereward']
        listact.append(a)
        obnew1 = np.concatenate((ob1[ob.shape[0]:], obnew))

        listob.append(obnew1)
        # (obnew, reward, done, _info) = env.step(a)
        #		print agent.evalQ(ob,a)-target,

        start = time.time()
        cost += agent.learn(ob1, a, obnew1, reward, 1. - 1. * done, agent.act(obnew1, episode))

        if render and (t % 20 == 0 or done):
            '''
			if agent.plotautoenc:
				listact1=np.array(listact)
				allactionsparse=np.zeros((listact1.shape[0],agent.n_out))
				allactionsparse[np.arange(listact1.shape[0]),listact1]=1.
				encob=agent.sess.run(agent.hiddencode,feed_dict={agent.sa:np.concatenate((np.array(listob)[:-1],allactionsparse),1) })
				Qlist= agent.evalQ(np.array(listob)[:-1],allactionsparse)
				Qlistn=(Qlist-np.min(Qlist))/(np.max(Qlist)-np.min(Qlist))
				ax[1].clear()
				ax[1].set_xlim(-1, 1)
				ax[1].set_ylim(-1, 1)
				ax[1].autoscale(False)
				#ax[1].quiver(encob[:,0],encob[:,1],encob[1:,0]-encob[:-1,0],encob[1:,1]-encob[:-1,1],color='black')
				ax[1].plot(encob[:,0],encob[:,1],color='black')
				ax[1].scatter(encob[:,0],encob[:,1],s=100,color=cm.rainbow(Qlistn.reshape(-1,)))
				plt.draw()
			'''
            print 'learn time', (time.time() - start) * 100., agent.maxq(
                ob1), reward  # agent.sess.run(agent.R, feed_dict={agent.x:ob1.reshape(1,-1)}),reward
        # print agent.sess.run(agent.Q, feed_dict={agent.x:ob.reshape(1,-1)})
        # for _ in xrange(19999):
        #	print 'testlearn',agent.testlearn(ob,a,obnew,100*reward,1.-1.*done),agent.evalQ(ob,a),100*reward
        # print agent.evalQ(ob,a)

        ob1 = obnew1
        total_rew += reward
        if render and t % 2 == 0:
            env.render()
        # time.sleep(0.)
        # print(a)
        if done: break
    return total_rew, t + 1, cost, listob, listact
