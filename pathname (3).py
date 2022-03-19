import os
import random
import shutil

import numpy as np
from PIL import Image
import shutil


# path1 = './datadml/kth2b/4/test/'
# path2 = './datadml/kth2b/4/train/'
# dirlist = os.listdir(path1)
# print(dirlist)
# for i in range(len(dirlist)):
#     newpath = './datadml/kth2b/4/train/' + dirlist[i]
#     isExists = os.path.exists(newpath)
#
#     # 判断结果
#     # 如果不存在则创建目录
#     # 创建目录操作函数
#     if not isExists:
#         os.makedirs(newpath)
#         print(newpath + ':创建成功')
#         # return True
#     else:
#         # 如果目录存在则不创建，并提示目录已存在
#         print(newpath + ':目录已存在')
floadername = ['sample_a','sample_b','sample_c','sample_d']
# for cg in range(len(dirlist)):
#     oname = path1 + dirlist[cg] + '/' + floadername[3] + '/'
#     nname = path2 + dirlist[cg] + '/'
#     shutil.move(oname,nname)


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

path1 = './datadml/kth2b/4/test/'
path2 = './datadml/kth2b/4/train/'
dirlist = os.listdir(path2)
print(dirlist)
# for i in range(len(dirlist)):
#     ooo = path2 + dirlist[i] + '/' + floadername[3] + '/'
#     ccc = path2 + dirlist[i] + '/'
#     for roots, dirs, files in os.walk(ooo):
#         print(files)
#         for imgname in files:
#             imgpath = ooo + imgname
#             # im = Image.open(imgpath)
#             # im = Image.open(imgpath).convert("RGB")
#             # im.save(newpath + imgname)
#             # os.remove(imgpath)  # 转移完后删除原图片
#             shutil.move(imgpath, ccc)
#     os.rmdir(ooo)
for i in range(len(dirlist)):
    ooo = path1 + dirlist[i] + '/' + floadername[2] + '/'
    ccc = path1 + dirlist[i] + '/'
    for roots, dirs, files in os.walk(ooo):
        print(files)
        for imgname in files:
            imgpath = ooo + imgname
            # im = Image.open(imgpath)
            # im = Image.open(imgpath).convert("RGB")
            # im.save(newpath + imgname)
            # os.remove(imgpath)  # 转移完后删除原图片
            shutil.move(imgpath, ccc)
    os.rmdir(ooo)

