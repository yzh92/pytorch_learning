from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

# writer.add_image()
# writer.add_scalar()
# y = x
# for i in range(100):
#     writer.add_scalar("y=x", i, i)
# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)
writer.close()

'''
tensorboard --logdir = logs 切换到当前目录
改端口：
tensorboard --logdir = logs --port = 6007
'''