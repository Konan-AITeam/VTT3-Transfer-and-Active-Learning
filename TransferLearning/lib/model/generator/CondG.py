from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import pdb



class _conditional_generator2(nn.Module):
    def __init__(self):
        super(_conditional_generator2, self).__init__()
        self.Blocks1 = self._make_layer(64)
        self.Blocks2 = self._make_layer(64)
        self.Blocks3 = self._make_layer(64)

    def _make_layer(self, ch):
        layers = []
        for i in range(3):
            layers.append(BasicBlock(ch, ch))

        return nn.Sequential(*layers)

    def forward(self, x):
        # h = y.size()[2]
        # w = y.size()[3]
        # avp = nn.AdaptiveAvgPool2d((h,w))
        x = self.Blocks1(x)
        x = self.Blocks2(x)
        x = self.Blocks3(x)

        return x


class _conditional_generator(nn.Module):
    def __init__(self):
        super(_conditional_generator, self).__init__()
        self.Bottleneck1 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Blocks1 = self._make_layer(64)
        self.Bottleneck2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.Blocks2 = self._make_layer(128)
        self.Bottleneck3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.Blocks3 = self._make_layer(256)
        # self.Bottleneck4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.Blocks4 = self._make_layer(256)
        self.Bottleneck5 = nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_layer(self, ch):
        layers = []
        for i in range(3):
            layers.append(BasicBlock(ch, ch))

        return nn.Sequential(*layers)

    def forward(self, x, y):
        h = y.size()[2]
        w = y.size()[3]
        avp = nn.AdaptiveAvgPool2d((h,w))
        x = self.Bottleneck1(x)
        x = self.Blocks1(x)
        x = self.Bottleneck2(x)
        x = self.Blocks2(x)
        x = self.Bottleneck3(x)
        x = self.Blocks3(x)
        # x = self.Bottleneck4(x)
        x = self.Blocks4(x)
        x = self.Bottleneck5(x)
        x = avp(x)

        return x


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out += residual

    return out
