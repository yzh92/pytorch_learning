import torch
from model_save import Model
#方式1->保存方式1，加载模型
# model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2，加载模型
model2 = torch.load('vgg16_method2.pth')
# print(model2)

# 陷阱导入

self_model = torch.load('self_define_model.pth')
print(self_model)