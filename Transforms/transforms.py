from torchvision import transforms
from PIL import Image
import cv2
from torch.utils.tensorboard import SummaryWriter
# 1.怎么使用transforms
# 获取图片
image_path = '../dataset_other_show/train/bees_image/17209602_fe5a5a746f.jpg'
img = Image.open(image_path)
print(type(img))

writer = SummaryWriter('logs')
# 转换为tensor类型图片
tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(img)
# print(img_tensor)

writer.add_image('Tensor_img',img_tensor)

writer.close()


# 2.为什么要把图片转换为tensor类型
# tensor类型中的属性包含了神经网络训练时所需要的理论基础，例如_backward_hooks, grad, grad_fn

#
cv_img = cv2.imread(image_path)

