
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np

def parse_cfg_file(cfgfile):
    """
    Param: Configuration file

    Returns a list of blocks for network block configurations from the config file.
    Blocks are represented through dictionaries
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces
    
    block = {}
    blocks = []
    for line in lines:
        if line[0] == '[':
            blocks.append(block)
            block = {} # Start new dict
            block['type'] = line[1:-1].rstrip()
        else:
            key,value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks = blocks[1:]
     

parse_cfg_file('cfg/yolov3.cfg')