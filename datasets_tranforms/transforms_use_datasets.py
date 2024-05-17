import torchvision
dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor])

train_set = torchvision.datasets.CIFAR10(root='./CIFAR10',train=True,download=True)
test_set = torchvision.datasets.CIFAR10(root='./CIFAR10',train=False,download=True)

print(train_set[0])
print(train_set.classes)
img,target = train_set[0]
print(img)
print(target)
print(train_set.classes[target])
img.show()