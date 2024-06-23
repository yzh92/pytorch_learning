import torch.optim
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# from model import *
# 加载数据集

train_data = torchvision.datasets.CIFAR10('./data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10('./data', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 定义训练设备
device = torch.device('cpu')

# 数据长度
train_data_size = len(train_data)
test_data_size = len(test_data)

print('train_data_size: {}'.format(train_data_size))
print('test_data_size: {}'.format(test_data_size))

# dataloader 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=64, drop_last=True)


# 构建模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    model = Model()
    input = torch.ones((64, 3, 32, 32))
    output = model(input)
    print(output.shape)
model = Model()
model = model.to(device)
# print(model)


# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 添加tensorboard
writer = SummaryWriter('./tensorboard')

# 训练的轮数
epoch = 10

# 最好测试精度变量
best_accuracy = 0

for i in range(epoch):

    print('----------------第{}轮训练开始----------------'.format(i + 1))
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = model(imgs)
        loss = loss_fn(output, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print('训练次数：{}，Loss:{}'.format(total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = model(imgs)
            loss = loss_fn(output, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy

    current_accuracy = total_accuracy / test_data_size
    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    # 保存最好正确率的模型权重
    if current_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'model_{}.pth'.format(i))

writer.close()