# -*- coding: utf-8 -*-
# @Time    : 2024/5/22 15:10
# @Author  : yzh
# @File    : nn_linear.py
# @Software: PyCharm
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10('./data',train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset,batch_size=64,drop_last=True)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear()

    def forward(self,input):
        output = self.linear(input)
        return output

model = Model()

for data in dataloader:
    imgs,targets = data
    imgs = torch.flatten(imgs)
    print(imgs.shape)