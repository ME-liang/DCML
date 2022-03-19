import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
import random
from PIL import Image



fileDir = "/home/cvnlp/LiChuanxiu/DataSets/BreaKHis/40/predict/malignant"

aimDir = "/home/cvnlp/LiChuanxiu/DataSets/BreaKHis/40/predict/malignant"

pathDir = os.listdir(fileDir)

sample = random.sample(pathDir, 100)

def rotation(root_path, img_name,angle): #旋转角度
    img = Image.open(os.path.join(root_path, img_name))
    rotation_img = img.rotate(angle)
    return rotation_img

for name in tqdm(sample):
    rotation180_saveName = name[:-4] + "-60R.png"
    rotation180_Img = rotation(fileDir, name, 60)
    rotation180_Img.save(os.path.join(aimDir, rotation180_saveName))