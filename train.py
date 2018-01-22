# -*- coding:utf-8 -*-
# Author: Xiangtai Li
# Training scripts for cifar-10
'''
Train Cifar10 with Pytorch
'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import argparse
import json
from models import *
from utils.logger import Logger
from utils.tools import update_hyper
from torch.autograd import Variable
from data_loader import Dataloader
from models.model import Model

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

use_cuda = torch.cuda.is_available() and (not cfg['no_cuda'])
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# DataLoader
print('==> Preparing data..')
trainloader,testloader = Dataloader(cfg['data_dir'],cfg['train_batch'],cfg['test_batch'])

# Net config
if cfg['resume']:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(cfg['check_dir']), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(cfg['check_dir']+'/'+cfg['model']+'.pth')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model from scratch')
    net = Model.model(cfg['model'])

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
if cfg['opt'] =='sgd':
    optimizer = optim.SGD(net.parameters(), lr=cfg["lr"], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
if cfg['opt'] =='adam':
    optimizer == optim.Adam(net.parameters(), lr=cfg["lr"], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
if cfg['opt'] == 'rmsprop':
    optimizer == optim.RMSprop(net.parameters(), lr=cfg["lr"], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        if batch_idx % 100 ==0:
            print("Training: Epoch: %d, batch: %d, Loss: %.3f , Acc: %.3f, Correct/Total: (%d/%d)"
                  % (epoch, batch_idx, train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        if batch_idx % 100 ==0:
            print("Testing: Epoch:%d, batch: %d, Loss: %.3f , Acc: %.3f, Correct/Total: (%d/%d)"
                  % (epoch, batch_idx, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(cfg['check_dir']):
            os.mkdir(cfg['check_dir'])
        torch.save(state, cfg['check_dir']+'/'+cfg['model']+'.pth')
        best_acc = acc

# train process
for epoch in range(start_epoch, cfg['epoch']):
    train(epoch)
    test(epoch)