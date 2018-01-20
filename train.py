'''
Train Cifar10 with Pytorch
'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils.tools import progress_bar
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch-size',default=32,type=int,help='batch size of the ')
parser.add_argument('--opt', type=str, default='sgd',choices=('sgd', 'adam', 'rmsprop'))
parser.add_argument('--no-cuda',default=False,help='whether use GPU')


args = parser.parse_args()


