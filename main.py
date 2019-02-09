import os
import sys
import argparse
import numpy as np
import torch

from torch import nn
from torch import optim
from torch.nn import functional as F
from imageio import imwrite

from cppn import load_args, cppn

if __name__ == '__main__':
    args = load_args()
    args.sample = True
    args.n = 10
    args.y_dim = 1080
    args.x_dim = 1080
    args.scale = 10
    args.net = 32
    args.c_dim = 1
    args.exp = 'test'
    cppn(args)
