from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        # 相对于程序的相对路径
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path) # 读取图片
        label = self.label_dir
        return img,label
    def __len__(self):
        return len(self.img_path)

root_dir = 'dataset/train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'
ants_dataset = MyData(root_dir,ants_label_dir)
bees_dataset = MyData(root_dir,bees_label_dir)

# 数据集集合
# + 运算符被解释为torch.utils.data.ConcatDataset类的实例创建
# + 号会自动调用父类Dataset类的__add__方法
train_dataset = ants_dataset + bees_dataset


ants_dataset[0]
