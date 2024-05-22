# -*- coding: utf-8 -*-
# @Time    : 2024/5/22 14:37
# @Author  : yzh
# @File    : nn_relu.py
# @Software: PyCharm
import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[-1,0.5],
                      [3,-2]])
input = torch.reshape(input,(-1,1,2,2))

dataset = torchvision.datasets.CIFAR10('./datasets_tranforms/CIFAR10',train=False,
                                       transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.relu = ReLU()
        # exchage sigmoid
        self.sigmoid = Sigmoid()
    def forward(self,input):
        # output = self.relu(input)
        output = self.sigmoid(input)
        return output

model = Model()
# output = model(input)
#
# print(output)
writer = SummaryWriter('logs')
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images('sigmoid_input',imgs,step)
    output = model(imgs)
    writer.add_images('sigmoid_output',output,step)
    step = step+1
writer.close()