from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
writer = SummaryWriter("logs")


# writer.add_scalar()

# y = x
# for i in range(100):
#     writer.add_scalar("y=x", i, i)

# y = 2x
# for i in range(100):
#     writer.add_scalar("y=2x", 2*i, i)
# writer.close()

# writer.add_image()
image_path = 'D:\Python project\pytorch_learning\dataset_other_show\\train\\ants_image\\5650366_e22b7e1065.jpg'
img_PIL = Image.open(image_path)
print(type(img_PIL))
img_array = np.array(img_PIL)
print(type(img_array))
writer.add_image('test',img_array,2,dataformats='HWC')

writer.close()

'''
tensorboard --logdir = logs 切换到当前目录
改端口：
tensorboard --logdir = logs --port = 6007
'''