# code : https://github.com/Aiolus-X/RefConv

import torch
import numpy as np
import math
import time
import random
import os
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm

import logging
from torch.nn.parameter import Parameter

class RepConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1,
                 map_k=3):
        super(RepConv, self).__init__()
        assert map_k <= kernel_size
        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))
        G = in_channels * out_channels // (groups ** 2)
        self.num_2d_kernels = out_channels * in_channels // groups
        self.kernel_size = kernel_size
        self.convmap = nn.Conv2d(in_channels=self.num_2d_kernels,
                                 out_channels=self.num_2d_kernels, kernel_size=map_k, stride=1, padding=map_k // 2,
                                 groups=G, bias=False)
        #nn.init.zeros_(self.convmap.weight)
        self.bias = None#nn.Parameter(torch.zeros(out_channels), requires_grad=True)     # must have a bias for identical initialization
        self.stride = stride
        self.groups = groups
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

    def forward(self, inputs):
        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
        kernel = self.weight + self.convmap(origin_weight).view(*self.origin_kernel_shape)
        return F.conv2d(inputs, kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.groups, bias=self.bias)


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        RepConv(inp, oup, kernel_size=3, stride=stride, padding=None, groups=1, map_k=3),
        #conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )
if __name__ == "__main__":

    # RepConv的输入和输出
    block = RepConv(128, 128, kernel_size=3, stride=1)
    input = torch.rand(3, 128, 64, 64)
    output = block(input)
    print(input.size(), '\n', output.size())


    # 由RepConv构成的卷积
    net = conv_bn(128, 128, stride=1)
    output2 = net(input)
    print(output2.size())
