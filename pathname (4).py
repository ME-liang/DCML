import os
import random
import shutil

import numpy as np
from PIL import Image
import shutil



for gh in range(10):
    imgnptrain = '/DATA/xqp/my02/datadml/dtd/' + str(gh+1) + '/train/'
    imgnpval = '/DATA/xqp/my02/datadml/dtd/' + str(gh+1) + '/val/'
    imgnptest = '/DATA/xqp/my02/datadml/dtd/' + str(gh+1) + '/test/'
    dirlist = os.listdir(imgnptrain)
    print(dirlist)
    print(len(dirlist))
    for i in range(len(dirlist)):
        newpath = '/DATA/xqp/my02/datadml/dtd/' + str(gh+1) + '/val/' + dirlist[i]
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
    for i in range(len(dirlist)):
        newpath = '/DATA/xqp/my02/datadml/dtd/' + str(gh+1) + '/test/' + dirlist[i]
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
    imgs_path1 = open('./datadml/dtd/labels/train' + str(gh+1) + '.txt').read().splitlines()
    imgs_path2 = open('./datadml/dtd/labels/val' + str(gh+1) + '.txt').read().splitlines()
    imgs_path3 = open('./datadml/dtd/labels/test' + str(gh+1) + '.txt').read().splitlines()
    # print('*'*50 + 'traindata' + '*'*50)
    # for i, img in enumerate(imgs_path1):
    #     newimgtrain = '/DATA/xqp/my02/datadml/dtd/1/train/' + img
    #     print("%d %s" % (i, img))
    #     print("%d %s" % (i, newimgtrain))
    # print("")
    # print('*'*50 + 'valdata' + '*'*50)
    for uu in range(len(dirlist)):
        for i, img in enumerate(imgs_path2):
            # print('*' * 50)
            # print(img[0:img.rfind('/')])
            # rr = dirlist[uu] in img
            if dirlist[uu] == img[0:img.rfind('/')]:
                print('it is val')
                newimgval = '/DATA/xqp/my02/datadml/dtd/' + str(gh+1) + '/train/' + img
                newx = imgnpval + dirlist[uu] + '/'
                print("%d %s" % (i, img))
                print("%d %s" % (i, newimgval))
                shutil.move(newimgval, newx)
            else:
                print('None')
        print("")

    for uu in range(len(dirlist)):
        for i, imgx in enumerate(imgs_path3):
            if dirlist[uu] == imgx[0:imgx.rfind('/')]:
                print('it is test')
                newimgtest = '/DATA/xqp/my02/datadml/dtd/' + str(gh+1) + '/train/' + imgx
                newx = imgnptest + dirlist[uu] + '/'
                print("%d %s" % (i, imgx))
                print("%d %s" % (i, newimgtest))
                shutil.move(newimgtest, newx)
            else:
                print('None')
        print("")
# print('*'*50 + 'testdata' + '*'*50)
# for i, img in enumerate(imgs_path3):
#     newimgtest = '/DATA/xqp/my02/datadml/dtd/1/train/' + img
#     print("%d %s" % (i, img))
#     print("%d %s" % (i, newimgtest))
# print("")


