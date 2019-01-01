import argparse
import logging
import ast
import pickle,os,sys

def rangeparameters(assigned):
    if "initial_learnrate" not in assigned:
        return "initial_learnrate", type(0.0), [0.00001, 0.1]
    if "discount" not in assigned:
        return "discount", type(0.0), [0.9, 0.995]
    if "policy" not in assigned:
        return "policy",type(True), [True,False]
    if "batch_size" not in assigned:
        return "batch_size",type(1),[5,10000]
    if assigned['policy']:
        if "lambda" not in assigned:
            return "lambda",type(0.0), [0.0,1.0]
        if "entropy" not in assigned:
            return "entropy", type(0.0), [0.0, 0.1]
        if "episodic" not in assigned:
            return "episodic", type(True), [True]

    if assigned['policy']==False:
        if "episodic" not in assigned:
            return "episodic", type(True), [True, False]
        if "lambda" not in assigned:
            return "lambda", type(0.0), [0.0, 0.0]

        if "copyQ" not in assigned:
            return "copyQ", type(True), [True, False]
        elif assigned['copyQ']:
            return "doubleQ", type(False), [False]
        else:
            return "doubleQ", type(True), [True, False]


def checkparams(params):
    if params['policy']:
        if params['copyQ']>0 or params['doubleQ']:
            return False,"policy with copyQ/doubleQ not allowed"
    return True,""

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def str2list(s):
    return ast.literal_eval(s)


def init_log(params):
    if params['logging']==0:
        loglevel = logging.DEBUG
    elif params['logging']==1:
        loglevel = logging.INFO
    elif params['logging']==2:
        loglevel = logging.WARNING
    elif params['logging']==3:
        loglevel = logging.ERROR
    logger = logging.getLogger(__name__)
    logger.setLevel(loglevel)
    logger.propagate=False
    if params["file"] is not None:
        handler = logging.FileHandler(params["file"]+'.log')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        ch = logging.StreamHandler()
        ch.setLevel(loglevel)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        ch.setFormatter(formatter)

        logger.addHandler(ch)
    return logger