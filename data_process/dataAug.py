from PIL import Image
from PIL import ImageEnhance
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def flip_horizontal(root_path,img_name):   #水平翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img


def flip_vertical(root_path,img_name):   #垂直翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return filp_img

def rotation(root_path, img_name,angle): #旋转角度
    img = Image.open(os.path.join(root_path, img_name))
    rotation_img = img.rotate(angle)
    return rotation_img


def imageResize(root_path, img_name, scale):  #缩放
    img = Image.open(os.path.join(root_path, img_name))
    width = int(img.size[0] * scale)
    height = int(img.size[1] * scale)
    img = img.resize((width, height), Image.ANTIALIAS)
    return img


imageDir= "/home/cvnlp/LiChuanxiu/BreaKHis_v1All"
flip_horizontalDir = "/home/cvnlp/LiChuanxiu/Augment/horizontal"
flip_verticalDir = "/home/cvnlp/LiChuanxiu/Augment/vertical"
rotation_180Dir = "/home/cvnlp/LiChuanxiu/Augment/60"
imageResizeDir = "/home/cvnlp/LiChuanxiu/Augment/0.8"


for name in tqdm(os.listdir(imageDir)):

    # flip_horizontal_saveName = name[:-4]+"-Hflip.png"
    # flip_horizontal_Img = flip_horizontal(imageDir,name)
    # flip_horizontal_Img.save(os.path.join(flip_horizontalDir,flip_horizontal_saveName))
    #
    # flip_vertical_saveName = name[:-4]+"-Vflip.png"
    # flip_vertical_Img = flip_vertical(imageDir,name)
    # flip_vertical_Img.save(os.path.join(flip_verticalDir,flip_vertical_saveName))

    rotation180_saveName = name[:-4] + "-60R.png"
    rotation180_Img = rotation(imageDir,name,60)
    rotation180_Img.save(os.path.join(rotation_180Dir,rotation180_saveName))

    # saveName = name[:-4]+"-resize.png"
    # Img = imageResize(imageDir,name,0.8)
    # Img.save(os.path.join(imageResizeDir,saveName))


