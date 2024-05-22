# -*- coding: utf-8 -*-
# @Time    : 2024/5/22 14:05
# @Author  : yzh
# @File    : nn_maxpool.py
# @Software: PyCharm
import torch
from torch import nn
from torch.nn import MaxPool2d

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]],dtype=torch.float)
input = torch.reshape(input,(-1,1,5,5))
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output = self.maxpool1(input)
        return output
model = Model()
output = model(input)
print(output)