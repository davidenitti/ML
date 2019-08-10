import argparse
import logging
import os
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


BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"


def formatter_message(message, use_color=True):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message


COLORS = {
    'WARNING': YELLOW,
    'INFO': BLUE,
    'DEBUG': WHITE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


def init_logger(log_path, level='INFO', mode='w'):
    """
    initialize the logger in the main script
    """
    level_console = getattr(logging,level)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    formatter = logging.Formatter('[%(name)s][%(levelname)s] %(message)s (%(filename)s:%(lineno)d)')  # %(asctime)s -
    if log_path:
        handler = logging.FileHandler(log_path, mode=mode)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    FORMAT = "[$BOLD%(name)-0s$RESET][%(levelname)-0s]  %(message)s ($BOLD%(filename)s$RESET:%(lineno)d)"
    color_format = formatter_message(FORMAT, True)
    formatter2 = ColoredFormatter(color_format)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level_console)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if log_path:
        root.addHandler(handler)
    root.addHandler(console_handler)