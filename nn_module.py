# -*- coding: utf-8 -*-
# @Time    : 2024/5/18 17:33
# @Author  : yzh
# @File    : nn_module.py
# @Software: PyCharm
import torch
from torch import nn


class Module(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,input):
        output = input + 1
        return output

model = Module()
x = torch.tensor(1.0)
output = model(x)
print(output)