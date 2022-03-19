import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


sourceDir = "/home/cvnlp/LiChuanxiu/BreaKHis_v1All"
saveDir ="/home/cvnlp/LiChuanxiu/Augment/random_crop"


for name in tqdm(os.listdir(sourceDir)):
    img = cv2.imread(os.path.join(sourceDir,name))
    crop_img = tf.random_crop(img, [400, 400, 3])
    sess = tf.InteractiveSession()
    saveName = name[:-4]+"-randomcrop.png"
    cv2.imwrite(os.path.join(saveDir,saveName), crop_img.eval())
    sess.close()