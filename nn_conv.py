# -*- coding: utf-8 -*-
# @Time    : 2024/5/18 18:19
# @Author  : yzh
# @File    : nn_conv.py
# @Software: PyCharm
import torch

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])

kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])

input = torch.reshape(input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))

output = torch.conv2d(input,kernel,stride=1)
print(output)
output2 = torch.conv2d(input,kernel,stride=2)
print(output2)

output3 = torch.conv2d(input,kernel,stride=1,padding=1)
print(output3)
