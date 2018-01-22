# -*- coding:utf-8 -*-
# Author: Xiangtai Li
# Training scripts for cifar-10
'''
Train Cifar10 with Pytorch
'''
from __future__ import print_function

import argparse
import json
from models import *
from utils.logger import Logger
from utils.tools import update_hyper
from cifar_classifier import CifarClassifier

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', default=True, action='store_true', help='resume from checkpoint')
parser.add_argument('--train_batch',default=64,type=int,help='batch size of training ')
parser.add_argument('--test_batch',default=64,type=int,help='batch size of test')
parser.add_argument('--opt', type=str, default='sgd',choices=('sgd', 'adam', 'rmsprop'))
parser.add_argument('--no_cuda',default=False,help='whether use GPU')
parser.add_argument('--data_dir',default="/home/lxt/data",help="Data floder")
parser.add_argument('--model',default="densenet121",help="CNN model")
parser.add_argument('--check_dir',default="/home/lxt/data",help="checkpoint for resume training")
parser.add_argument('--momentum',default=0.9, help="set momentum of optimizer")
parser.add_argument('--weight_decay',default=5e-4, help="set weight decay of optimizer")
parser.add_argument('--epoch',default=20,help="training epoch of the process")
args = parser.parse_args()
# load default parameters
cfg = json.load(open("./config/parameters.json","r"))
# adjust parameters by command line
cfg = update_hyper(args,cfg)

if __name__ == '__main__':
    clf = CifarClassifier(cfg)
    clf.train()