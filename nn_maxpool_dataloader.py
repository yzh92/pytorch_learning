# -*- coding: utf-8 -*-
# @Time    : 2024/5/22 14:15
# @Author  : yzh
# @File    : nn_maxpool_dataloader.py
# @Software: PyCharm
import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(',/datasets_tranforms/CIFAR10', train=True,
                                       transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset,batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = MaxPool2d(kernel_size=3)

    def forward(self,input):
        output = self.maxpool(input)
        return output

model = Model()

writer = SummaryWriter('logs')
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images('input2',imgs,step)
    output = model(imgs)
    writer.add_images('output2',output,step)
    step = step+1
writer.close()
