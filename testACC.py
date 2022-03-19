import torch
import torch.optim as optim
import models
from data_loader import get_test_loader, get_train_loader
from config import get_config
from utils import accuracy, AverageMeter
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from utils import loader_model

config, unprased = get_config()

modelpath = '/media/cvnlp/3b670053-8188-42b6-a0aa-7390926a3303/home/cvnlp/LiChuanxiu/实验/googlenet/multi/400/googlenet_multi_e50_lr001_400X_model_best.pth.tar'
# self.model = getModel()
# self.model = models.get_vgg16(self.num_classes, self.feature_extract,self.use_pretrained, self.paramseed)
m, _, __ = loader_model(8, 'googlenet', modelpath)

test_dataset = get_test_loader(config.predictdata_dir,config.batch_size,config.input_size,config.num_workers)

losses = AverageMeter()
accs = AverageMeter()


m.cuda()
m.eval()


p = []
l = []
for i, (images, labels) in enumerate(test_dataset):

    images, labels = images.cuda(), labels.cuda()
    images, labels = Variable(images), Variable(labels)
    cpu_label = labels.cpu().detach().numpy()
    for e in(cpu_label):
        l.append(e)


    outputs = m(images)


    prec = accuracy(outputs, labels)
    accs.update(prec, images.size()[0])

    out = F.log_softmax(outputs)
    score = out.cpu().detach().numpy()
    for e in(score):
        p.append(e[1])

print(accs.avg)

