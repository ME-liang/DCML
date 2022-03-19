import os
import random

import numpy as np
from PIL import Image

path = "E:/achievement/dataset/4/train"
# path = "C:/Users/wurenzhong/Desktop/data"
# filelist = []
# for root, dirs, files in os.walk(path):
#     print(dirs)

dirlist = os.listdir(path)
print(dirlist)
print(len(dirlist))
for i in range(len(dirlist)):
    newpath = 'E:/achievement/dataset/4/test/' + dirlist[i]
    isExists = os.path.exists(newpath)

    # 判断结果
    # 如果不存在则创建目录
    # 创建目录操作函数
    if not isExists:
        os.makedirs(newpath)
        print(newpath + ':创建成功')
        # return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(newpath + ':目录已存在')
        # return False
    # os.mkdir(newpath)
    # print(dirs)
    # print(files)
    # for file in files:
    #     file = os.path.join(root, file)
        # with open('E:/dataest/FI/fff.txt', 'a') as fileobject:
        #     fileobject.writelines(file+'\n')
# for root, dirs, files in os.walk(path):
#     print(files)
# 只获取当前路径下文件名,不获取文件夹中文件名


import os

# def file_name(file_dir):
#     L=[]
#     for root, dirs, files in os.walk(file_dir):
#         for file in files:
#             if os.path.splitext(file)[1] == '.png':  # 想要保存的文件格式
#                 L.append(os.path.join(root, file))
#         return L
# newfilepath = file_name('E:/achievement/dataset/1/train/cotton')
# print(newfilepath)
# print(len(newfilepath))
# 将路径拆分为文件名+扩展名
def transferPictures(nowpath, newpath):
    # 将文件夹下的不同类别的文件夹中的部分图片转移到另一个文件夹下的相同类别的文件夹下，并删除原文件夹中的相应图片（类似于剪切）
    for roots, dirs, files in os.walk(nowpath):

        fnum = len(files) // 4  # 计算一半数量
        print(fnum)
        rdom_files = random.sample(files, fnum)  # 随机选一半数量的图片
        print(rdom_files)

        for imgname in rdom_files:
            imgpath = nowpath + imgname
            im = Image.open(imgpath)
            im.save(newpath + imgname)
            os.remove(imgpath)  # 转移完后删除原图片





for i in range(len(dirlist)):
    nowpath = 'E:/achievement/dataset/4/train/' + dirlist[i] + '/'
    repath = 'E:/achievement/dataset/4/test/' + dirlist[i] + '/'
    transferPictures(nowpath, repath)

# filename='data_batch_1.mat';
# data=[];
# labels=[];
# for i=1:5
#     file=matfile(filename);
#     data=[data;file.data];
#     labels=[labels;file.labels];
#     filename(12)=int2str(i+1);
# end
# save('train.mat','data','labels')
