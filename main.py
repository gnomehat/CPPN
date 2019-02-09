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
    cppn(args)
