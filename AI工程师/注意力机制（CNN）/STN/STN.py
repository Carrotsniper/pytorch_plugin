# code: https://github.com/tensorflow/models/blob/master/transformer/cluttered_mnist.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        # 定义本地化网络，用于估计空间变换的参数
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),  # 输入通道数为 1，输出通道数为 8，卷积核大小为 7
            nn.MaxPool2d(2, stride=2),  # 最大池化层，核大小为 2，步长为 2
            nn.ReLU(True),  # ReLU 激活函数
            nn.Conv2d(8, 10, kernel_size=5),  # 输入通道数为 8，输出通道数为 10，卷积核大小为 5
            nn.MaxPool2d(2, stride=2),  # 最大池化层，核大小为 2，步长为 2
            nn.ReLU(True)  # ReLU 激活函数
        )
        # 定义空间变换网络，用于预测空间变换的参数
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),  # 全连接层，输入维度为 10 * 3 * 3，输出维度为 32
            nn.ReLU(True),  # ReLU 激活函数
            nn.Linear(32, 3 * 2)  # 全连接层，输入维度为 32，输出维度为 3 * 2
        )
        # 初始化空间变换网络的权重和偏置
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        # 使用本地化网络对输入图像进行特征提取
        xs = self.localization(x)
        # 将特征张量展开成一维张量
        xs = xs.view(-1, 10 * 3 * 3)
        # 使用空间变换网络预测空间变换的参数
        theta = self.fc_loc(xs)
        # 将一维张量转换成二维张量，用于执行仿射变换
        theta = theta.view(-1, 2, 3)
        # 使用仿射变换对输入图像进行空间变换
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.stn = STN()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # 使用 STN 对输入图像进行空间变换
        x = self.stn(x)
        # 经过卷积和池化层处理
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

if __name__ == "__main__":

    block = STN()
    input = torch.rand(4, 1, 28, 28)                # 这里有问题
    output = block(input)
    print(input.size(), '\n', output.size())