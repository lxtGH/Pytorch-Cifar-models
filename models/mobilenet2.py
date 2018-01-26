#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiangtai
# Mobile net  V2 implementation
'''
MobileNetv2 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segment"
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self,inplanes,outplanes,stride=1,expansion=1,first=False):
        super(Bottleneck, self).__init__()
        self.point_conv1 = nn.Conv2d(inplanes,expansion*inplanes,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(expansion*inplanes)
        self.depth_conv = nn.Conv2d(expansion*inplanes,expansion*inplanes,3,stride=stride,padding=1,bias=False
                                    ,groups=expansion*inplanes)
        self.bn2 = nn.BatchNorm2d(expansion*inplanes)
        self.point_conv2 = nn.Conv2d(expansion*inplanes,outplanes,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.first = first
        self.model = nn.Sequential(
            self.point_conv1,self.bn1, self.relu,
            self.depth_conv,self.bn2, self.relu,
            self.point_conv2,self.bn3
        )
    def forward(self, x):
        res = x
        print (x.size())
        if self.first == True: #first layer to change the channel
            res = 0
        out = self.point_conv1(x)
        out = self.bn1(out)

        out = self.depth_conv(out)
        out = self.bn2(out)

        out = self.point_conv2(out)
        out = self.bn3(out)

        out += res
        return out


class MobileNetv2(nn.Module):
    # bottleneck configuration  (number,stride,expansion)
    def __init__(self,bottle_neck,num_classes):

        super(MobileNetv2, self).__init__()
        self.bottle_neck = bottle_neck
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=2,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(bottle_neck, 32 , 16, 1, 1, 1)
        self.layer2 = self._make_layer(bottle_neck, 16, 24, 2, 2, 6)
        self.layer3 = self._make_layer(bottle_neck, 24, 32, 3, 2, 6)
        self.layer4 = self._make_layer(bottle_neck, 32, 64, 4, 2, 6)
        self.layer5 = self._make_layer(bottle_neck, 64, 96, 3, 1, 6)
        self.layer6 = self._make_layer(bottle_neck, 96, 160, 3, 2, 6)
        self.layer7 = self._make_layer(bottle_neck, 160,320, 1, 1, 6)

        self.layer8 = nn.Sequential(
            nn.Conv2d(320,1280,kernel_size=1,stride=1,bias=False),
            nn.AvgPool2d(7,stride=1),
            nn.Conv2d(1280,num_classes,kernel_size=1,stride=1,bias=False)
        )
    def _make_layer(self,bottle_neck,inplanes,outplanes,block_num,stride,expansion):
        layers = []

        first_bottle = True #
        layers.append(bottle_neck(inplanes,outplanes,stride,expansion,first_bottle))
        for i in range(1,block_num):
            # print ("bottle:",i)
            layers.append(bottle_neck(outplanes,outplanes,stride=1,expansion=expansion))

        return nn.Sequential(*layers)
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        out = out.view(out.size(0),-1)

        return out


model = MobileNetv2(Bottleneck,10)
print (model)
# test
x = torch.autograd.Variable(torch.randn(20, 3, 224, 224))

print (model(x))
