# -*- coding: utf-8 -*-

import os
import shutil

os.mkdir('./data')
os.mkdir('./data/train')
os.mkdir('./data/val')
os.mkdir('./data/train/dog')
os.mkdir('./data/train/cat')
os.mkdir('./data/val/dog')
os.mkdir('./data/val/cat')


file_list = os.listdir('./train')

dog_file = list(filter(lambda x:x[:3]=='dog', file_list))
cat_file = list(filter(lambda x:x[:3]=='cat', file_list))

for i in range(len(dog_file)):
    ori_path = os.path.join('./train', dog_file[i])
    if i < 0.9*len(dog_file):
        obj_path = os.path.join('./data/train/dog', dog_file[i])
    else:
        obj_path = os.path.join('./data/val/dog', dog_file[i])
    shutil.move(ori_path, obj_path)

for i in range(len(cat_file)):
    ori_path = os.path.join('./train', cat_file[i])
    if i < 0.9*len(cat_file):
        obj_path = os.path.join('./data/train/cat', cat_file[i])
    else:
        obj_path = os.path.join('./data/val/cat', cat_file[i])
    shutil.move(ori_path, obj_path)




