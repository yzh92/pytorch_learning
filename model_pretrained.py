import torchvision
from torch import nn
from torchvision.models.vgg import VGG16_Weights
# train_data = torchvision.datasets.ImageNet('./data',split='train',download = True,transform=torchvision.transforms.ToTensor())


vgg16_false = torchvision.models.vgg16(progress = True)
vgg16_true = torchvision.models.vgg16(weights = VGG16_Weights.DEFAULT,progress = True)

print('ok')

print(vgg16_true)
# nn.Conv2d()

train_data = torchvision.datasets.CIFAR10('./data',train=True,transform=torchvision.transforms.ToTensor(),
                                          download=True)


vgg16_true.classifier.add_module('add_linear',nn.Linear(100,10))

print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096,10)
print(vgg16_false)

