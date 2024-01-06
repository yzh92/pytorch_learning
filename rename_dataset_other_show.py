import os
# 将每个图片的名字在label文件加下生成对应的txt文件
root_dir = 'dataset_other_show/train'
target_dir = 'bees_image'

image_path =os.listdir(os.path.join(root_dir,target_dir))
label = 'bees'

out_dir = 'bees_label'
for item in image_path:
    # print(item)
    file_name = item.split('.')[0]
    with open(os.path.join(root_dir,out_dir,"{}.txt".format(file_name)),'w') as f:
        f.write(label)
