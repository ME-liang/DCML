import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm


sourceDir = "/home/cvnlp/LiChuanxiu/DataSets/BreaKHis/40/predict/benign"

s = 'SOB_B_.*.-40-.*.-60R.png'

for name in os.listdir(sourceDir):
    if re.compile(s).match(name):
       os.remove(os.path.join(sourceDir,name))