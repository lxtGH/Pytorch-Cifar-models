from data_loader import Dataloader
import os
import torch
import torch.cuda
import torch.backends.cudnn as cudnn
from models.model import Model
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class CifarClassifier(object):
    def __init__(self,cfg):
        super(CifarClassifier, self).__init__()
        self.net = None
        self.cfg = cfg
        self.trainloader, self.testloader = Dataloader(cfg['data_dir'], cfg['train_batch'], cfg['test_batch'])
        self.use_cuda = torch.cuda.is_available() and (not cfg['no_cuda'])
        self.best_acc = 0  # best test accuracy
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        # Net Config
        if cfg['resume']:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir(cfg['check_dir']), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(cfg['check_dir'] + '/' + cfg['model'] + '.pth')
            self.net = checkpoint['net']
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']
        else:
            print('==> Building model from scratch')
            self.net = Model.model(cfg['model'])
        # Cuda Config
        if self.use_cuda:
            self.net.cuda()
            self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        # Optimizer Config
        if cfg['opt'] == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(), lr=cfg["lr"], momentum=cfg['momentum'],
                                  weight_decay=cfg['weight_decay'])
        if cfg['opt'] == 'adam':
            self.optimizer == optim.Adam(self.net.parameters(), lr=cfg["lr"], momentum=cfg['momentum'],
                                    weight_decay=cfg['weight_decay'])
        if cfg['opt'] == 'rmsprop':
            self.optimizer == optim.RMSprop(self.net.parameters(), lr=cfg["lr"], momentum=cfg['momentum'],
                                       weight_decay=cfg['weight_decay'])
        if cfg['opt'] == 'adagrad':
            self.optimizer == optim.Adagrad(self.net.parameters(),lr=cfg["lr"], momentum=cfg['momentum'],
                                            weight_decay=cfg['weight_decay'])
        # Loss Config
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self,epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            self.optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            if batch_idx % 100 == 0:
                print("Training: Epoch: %d, batch: %d, Loss: %.3f , Acc: %.3f, Correct/Total: (%d/%d)"
                      % (epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


    def test_epoch(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            if batch_idx % 100 == 0:
                print("Testing: Epoch:%d, batch: %d, Loss: %.3f , Acc: %.3f, Correct/Total: (%d/%d)"
                      % (epoch, batch_idx, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        # Save checkpoint.
        acc = 100. * correct / total
        if acc > self.best_acc:
            print('Saving..')
            state = {
                'net': self.net.module if self.use_cuda else self.net,
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(self.cfg['check_dir']):
                os.mkdir(self.cfg['check_dir'])
            torch.save(state, self.cfg['check_dir'] + '/' + self.cfg['model'] + '.pth')
            best_acc = acc

    def train(self):
        for epoch in range(self.start_epoch, self.cfg['epoch']):
            self.train_epoch(epoch)
            self.test_epoch(epoch)

    def test(self):
        for epoch in range(self.start_epoch, self.cfg['epoch']):
            self.test_epoch(epoch)
