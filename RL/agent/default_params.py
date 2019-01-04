def get_default(nameenv):
    if nameenv == 'CartPole-v0':
        params = {
            "loss": "mse",
            "optimizer": "adam",
            "terminal_life":False,
            "start_episode":0,
            'transition_net': False,
            "priority_memory": False,
            "transition_weight": 0.01,
            "normalize": False,
            "memsize": 20000, # number of past experience tuples to store for learning
            "randstart": 50,
            "scaleobs": 1,
            "policy":True, # policy gradient
            "discounted_policy_grad": False, # use discounted policy gradient formula
            'doubleQ':False, # double Q learning (used when policy=False)
            "testeps": 0.,
            "copyQ": -1, # >=1 when policy=False
            "scalereward": 1,
            "limitreward": None,
            "batch_norm": False,
            "probupdate": 1.0,
            "lambda": .97, # 0.0 when policy=False
            "past": 0,
            "entropy":0.001,
            "eps": 0.5,  # Epsilon in epsilon greedy policies
            "mineps": 0.01,
            "decay": 0.995,  # Epsilon decay in epsilon greedy policies
            "initial_learnrate": 0.0007,
            "decay_learnrate": 0.9999,
            "discount": 0.99,
            "batch_size": 256,
            "episodic": True,
            "hiddenlayers": [10],
            "sharedlayers":[60],
            "regularization": 0.0,
            "file": None,
            "activation": 'elu', # tanh, sigmoid, relu, elu
            "conv":False,
            "scaling": 'none',
            "seed": -1,
            "threads": 0
            }
    elif nameenv == 'Acrobot-v1':
        params = {
            'transition_net':False,
            "priority_memory": False,
            "transition_weight":0.01,
            "normalize":True,
            "loss":"clipmse",
            "optimizer":"adam",
            "initial_learnrate": 0.0003,
            "decay_learnrate": 1,
            "lambda": 0.,
            "eps": 0.5,  # Epsilon in epsilon greedy policies
            "mineps": 0.05,
            "testeps": 0.00,
            "decay": 0.996,  # Epsilon decay in epsilon greedy policies
            "batch_norm" : False,
            "memsize": 150000,
            "randstart": 100,
            "policy":False,
            "discounted_policy_grad": False,
            'doubleQ':False,
            "copyQ": -1,
            "scalereward": 1,
            "scaleobs": 1.,
            "limitreward": None,
            "probupdate": 1,
            "entropy":0.01,
            "past": 0,
            "discount": 0.99,
            "batch_size": 64,
            "episodic": True,
            "hiddenlayers": [],
            "sharedlayers": [32],
            "regularization": 0.0000,
            "file": None,
            "activation": 'leaky_relu',
            "conv":False,
            "scaling": 'none',
            "seed": -1,
            "terminal_life": False,
            "start_episode": 0,
            "threads": 0
            }
    elif nameenv == 'LunarLander-v2':
        params = {
            'transition_net':False,
            "normalize":False,
            "priority_memory":True,
            "loss":"clipmse",
            "optimizer":"adam",
            "initial_learnrate": 0.00025,
            "decay_learnrate": 1,
            "lambda": 0.,
            "eps": 0.6,  # Epsilon in epsilon greedy policies
            "mineps": 0.05,
            "testeps": 0.00,
            "decay": 0.9955,  # Epsilon decay in epsilon greedy policies
            "batch_norm" : False,
            "memsize": 100000,
            "randstart": 100,
            "policy":False,
            "discounted_policy_grad": False,
            'doubleQ':False,
            "copyQ": -500,
            "scalereward": 1,
            "scaleobs": 1.,
            "limitreward": None,
            "probupdate": 1,
            "entropy":0.01,
            "past": 0,
            "discount": 0.99,
            "batch_size": 64,
            "episodic": True,
            "hiddenlayers": [],
            "sharedlayers": [256],
            "regularization": 0.000000,
            "file": None,
            "activation": 'leaky_relu',
            "conv":False,
            "scaling": 'none',
            "seed": 3,
            "start_episode": 0,
            "terminal_life": False,
            "threads": 0
        }
    elif nameenv == 'Breakout-v0':
        params = {
            "terminal_life": True,
            "loss":"clipmse",
            "start_episode": 0,
            "memsize": 1000000,
            "randstart": 5000,
            "policy":False,
            "batch_norm": False,
            "discounted_policy_grad":False,
            "scalereward": 0.1,
            "scaleobs": 1./255.,
            "limitreward": [-1., 1.],
            'doubleQ':False,
            "copyQ":-1,
            "probupdate": 1.,
            "lambda": 0.,
            "entropy": 0.01,
            "episodic":False,
            "past": 3,
            "eps":  .99,  # Epsilon in epsilon greedy policies
            "mineps": 0.1,
            "testeps": 0.05,
            "decay": 0.9996,  # Epsilon decay in epsilon greedy policies
            "initial_learnrate": 0.00015,
            "ratio_policy_learnrate": 1,
            "final_learnrate": 0.0001,
            "decay_learnrate": 1,
            "discount": 0.98,
            "batch_size": 32,
            "hiddenlayers": [256],
            "regularization": [0.0000000,0.0000000,0.0000000],
            "activation":'elu',
            "convlayers": [[8, 8, 4, 16], [4, 4, 2, 32]],
            "scaling":'crop',
            "dimdobs":(84,84,1),
            "file": None,
            "conv":True,
            "seed": -1,
            "threads": 0,
            "initializer": 'fixed',
            "sharedlayers": []}
    elif nameenv == 'Pong-v0':
        params = {
            "memsize": 2000,
            "storememory": False,
            "memoryfile": "/tmp/Pong2",
            "memorydir": "/tmp/memory",
            "replace": [],#"copyQ","memsize","initial_learnrate","decay_learnrate"],#"discount","episodic","activation","file","storememory","momentum","scalereward","probupdate","lambda","initial_learnrate","eps","batch_size","iterations","memsize"],
            "scalereward": 1.,
            "copyQ": 10,
            "probupdate": .02,
            "lambda": 0.1,
            "episodic":False,
            "past": 1,
            "eps": 1,  # Epsilon in epsilon greedy policies
            "decay": 0.999,  # Epsilon decay in epsilon greedy policies
            "initial_learnrate": 0.0001,
            "decay_learnrate": 0.9999,
            "discount": 0.99,
            "batch_size": 32,
            "iterations": 1,
            "hiddenlayers": [9,9,450],
            "regularization": [0.000000001],
            "activation":'tanh',
            "momentum": 0.8,
            "file": "pong-model_new15",
            "conv":True,
            "diffstate":False,
            "seed": 1}
    else:
        raise NotImplemented
    return params