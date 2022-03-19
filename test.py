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


device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
config, unprased = get_config()
path1 = "d:/PycharmProjects/resnet50_multi_e50_lr001_400X_model_best.pth.tar"
path2 = "d:/PycharmProjects/googlenet_multi_e50_lr001_400X_model_best.pth.tar"
checkpoint = torch.load(path1,map_location='cpu')
checkpoint2 = torch.load(path2,map_location='cpu')
model = models.get_resnet50(8, False, False, 1)
model2 = models.get_googlenet(8, False, False, 1)

model.load_state_dict(checkpoint['model_state'])
model2.load_state_dict(checkpoint2['model_state'])

test_dataset = get_test_loader(config.predictdata_dir,config.batch_size,config.input_size,config.num_workers)

losses = AverageMeter()
accs = AverageMeter()

# model.to(device1)
# model.eval()

p = []
l = []
for i, (images, labels) in enumerate(test_dataset):

    # images, labels = images.to(device1), labels.to(device1)
    images, labels = Variable(images), Variable(labels)
    cpu_label = labels.cpu().detach().numpy()
    for e in(cpu_label):
        l.append(e)


    outputs = model(images)

    print(outputs.argmax(dim=1))
    prec = accuracy(outputs, labels)
    accs.update(prec, images.size()[0])

    out = F.log_softmax(outputs)

    score = out.cpu().detach().numpy()
    for e in(score):
        p.append(e[1])

print(accs.avg)

#
# def loader_model(path):
#     checkpoint = torch.load(path)
#     model = models.get_resnet34(2, False, False, 1)
#     model.load_state_dict(checkpoint['model_state'])
#     optimizer = optim.SGD(model.parameters(), lr=0.001)
#     optimizer.load_state_dict(checkpoint['optim_state'])
#     epoch = checkpoint['epoch']
#     return model,optimizer,epoch
#
#
# config, unprased = get_config()
# #
# path = "./ckpt/resnet34_nopre1_model_best_DML.pth.tar"
# path2 = "./ckpt/resnet34_nopre2_model_best_DML.pth.tar"
# checkpoint = torch.load(path)
# checkpoint2 = torch.load(path2)
#
# model = models.get_resnet34(2, False, False, 1)
# model2 = models.get_googlenet(2, False, False, 1)
#
# optimizer = optim.SGD(model.parameters(), lr=0.001)
# model.load_state_dict(checkpoint['model_state'])
# model2.load_state_dict(checkpoint2['model_state'])
#
#
# optimizer.load_state_dict(checkpoint['optim_state'])
# epoch = checkpoint['epoch']


#
# print(list(model.children())[0])
# print(list(model.children())[1])
# print('*************************')
# for i,n in enumerate(list(model2.children())):
#     print(i)
#     print(n)


class Net(nn.Module):
    def __init__(self , model,model2):
        super(Net, self).__init__()
        self.layer1 = list(model.children())[0]
        self.layer2 = list(model.children())[1]
        self.layer3 = list(model.children())[2]
        self.layer4 = list(model.children())[3]
        self.layer5 = list(model.children())[4]
        self.layer6 = list(model.children())[5]
        self.layer7 = list(model.children())[6]
        self.layer8 = list(model.children())[7]
        self.layer9 = list(model.children())[8]

        self.LAYER1 = list(model2.children())[0]
        self.LAYER2 = list(model2.children())[1]
        self.LAYER3 = list(model2.children())[2]
        self.LAYER4 = list(model2.children())[3]
        self.LAYER5 = list(model2.children())[4]
        self.LAYER6 = list(model2.children())[5]
        self.LAYER7 = list(model2.children())[6]
        self.LAYER8 = list(model2.children())[7]
        self.LAYER9 = list(model2.children())[8]
        self.LAYER10 = list(model2.children())[9]
        self.LAYER11 = list(model2.children())[10]
        self.LAYER12 = list(model2.children())[11]
        self.LAYER13 = list(model2.children())[12]

        self.LAYER14 = list(model2.children())[13]
        self.LAYER15 = list(model2.children())[14]
        self.LAYER16 = list(model2.children())[15]
        self.LAYER17 = list(model2.children())[18]

        self.LAYER18 = list(model2.children())[19]
        self.LAYER19 = list(model2.children())[20]

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(1536, 832)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(832, 832)
        self.fc3 = nn.Linear(832, 2)



    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = x.reshape(x.size(0), -1)

        y = self.LAYER1(input)
        y = self.LAYER2(y)
        y = self.LAYER3(y)
        y = self.LAYER4(y)
        y = self.LAYER5(y)
        y = self.LAYER6(y)
        y = self.LAYER7(y)
        y = self.LAYER8(y)
        y = self.LAYER9(y)
        y = self.LAYER10(y)
        y = self.LAYER11(y)
        y = self.LAYER12(y)
        y = self.LAYER13(y)
        y = self.LAYER14(y)
        y = self.LAYER15(y)
        y = self.LAYER16(y)
        y = self.LAYER17(y)
        y = y.view(y.size(0), -1)
        output = torch.cat((x, y), 1)
        output = self.dropout(output)
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.relu(output)
        output = self.fc3(output)

        return output

# for index,p in enumerate(model.parameters()):
#     p.requires_grad = False
#
# for index,p in enumerate(model2.parameters()):
#     p.requires_grad = False
#
# model = Net(model,model2)

def getModel():
    path = "alexnet/resnet34_nopre1_model_best_DML.pth.tar"
    path2 = "./ckpt/resnet34_nopre2_model_best_DML.pth.tar"
    checkpoint = torch.load(path)
    checkpoint2 = torch.load(path2)

    model = models.get_resnet34(2, False, False, 1)
    model2 = models.get_googlenet(2, False, False, 1)
    model.load_state_dict(checkpoint['model_state'])
    model2.load_state_dict(checkpoint2['model_state'])

    for index, p in enumerate(model.parameters()):
        p.requires_grad = False

    for index, p in enumerate(model2.parameters()):
        p.requires_grad = False

    model = Net(model, model2)
    return model


# for index,p in enumerate(model.parameters()):
#     print(p.requires_grad)

# test_dataset = get_test_loader(config.predictdata_dir,config.batch_size,config.input_size,config.num_workers)

# losses = AverageMeter()
# accs = AverageMeter()
#
# model.to(device1)
# model.eval()
#
# model1.cuda()
# model1.eval()
#
#
# p = []
# l = []
# for i, (images, labels) in enumerate(test_dataset):
#
#     images, labels = images.to(device1), labels.to(device1)
#     images, labels = Variable(images), Variable(labels)
#     cpu_label = labels.cpu().detach().numpy()
#     for e in(cpu_label):
#         l.append(e)
#
#
#     outputs = model(images)
#     print(outputs.shape)
#
#
#     prec = accuracy(outputs, labels)
#     accs.update(prec, images.size()[0])
#
#     out = F.log_softmax(outputs)
#     score = out.cpu().detach().numpy()
#     for e in(score):
#         p.append(e[1])
#
# print(accs.avg)
















# print(model)
# class Net(nn.Module):
#     def __init__(self , model):
#         super(Net, self).__init__()
#         model.children()
#     def forward(self, x):
#         x = self.resnet_layer(x)
#         return x
#
# # MODEL = Net(model)
#
# print(list(model.children()))

# test_dataset = get_test_loader(config.predictdata_dir,config.batch_size,config.input_size,config.num_workers)
#
# # losses = AverageMeter()
# accs = AverageMeter()
#
# MODEL.to(device1)
# MODEL.eval()
# #
# # model1.cuda()
# # model1.eval()
# #
# #
# p = []
# l = []
# for i, (images, labels) in enumerate(test_dataset):
#
#     images, labels = images.to(device1), labels.to(device1)
#     images, labels = Variable(images), Variable(labels)
#     cpu_label = labels.cpu().detach().numpy()
#     for e in(cpu_label):
#         l.append(e)
#
#     outputs = MODEL(images)

#     prec = accuracy(outputs, labels)
#     accs.update(prec, images.size()[0])
#
#     out = F.log_softmax(outputs)
#     score = out.cpu().detach().numpy()
#     for e in(score):
#         p.append(e[1])
#
# print(accs.avg)
#
