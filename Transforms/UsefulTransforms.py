from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# tensorboard 配置
writer = SummaryWriter('logs')

image_path = '../images/pytorch.jpg'
img = Image.open(image_path)
# print(img)

# Totensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image('ToTensor',img_tensor,4)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm)



# Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
# img PIL -> resize -> img PIL
img_resize = trans_resize(img)
# img_resize PIL -> totensor -> img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize",img_resize,0)
# print(img_resize)

# Compose
trans_resize_2 = transforms.Resize(100)
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image('Resize',img_resize_2,2)


# RandomCrop
trans_random_crop = transforms.RandomCrop((96,100))
trans_compose_2 = transforms.Compose([trans_random_crop,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCropHW",img_crop,i)


writer.close()


