import os, random, shutil

fileDir = "/home/cvnlp/LiChuanxiu/DataSets/BreaKHis/400/train/benign/"
tarDir = "/home/cvnlp/LiChuanxiu/DataSets/BreaKHis/400/predict/benign/"


pathDir = os.listdir(fileDir)

sample = random.sample(pathDir, 1059)


for name in sample:
    shutil.copyfile(fileDir+name, tarDir+name)
    os.remove(fileDir+name)