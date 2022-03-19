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

def loader_model(path):
    checkpoint = torch.load(path)
    model = models.get_resnet34(2, False, False, 1)
    model.load_state_dict(checkpoint['model_state'])
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optim_state'])
    epoch = checkpoint['epoch']
    return model,optimizer,epoch


config, unparsed = get_config()
#
path = "alexnet/resnet34_nopre2_model_best_DML.pth.tar"
#
checkpoint = torch.load(path)

model = models.get_googlenet(2, False, False, 1)
optimizer = optim.SGD(model.parameters(), lr=0.001)
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optim_state'])
epoch = checkpoint['epoch']
# print(epoch)
# print(optimizer.param_groups[0]['momentum'])
#
#
#
# checkpoint1 = torch.load(path)
#
# model1 = models.get_resnet34(2, False, False, 1)
# optimizer1 = optim.SGD(model1.parameters(), lr=0.001)
# model1.load_state_dict(checkpoint1['model_state'])
# optimizer1.load_state_dict(checkpoint1['optim_state'])
# epoch1 = checkpoint1['epoch']
# print(epoch1)
# print(optimizer1.param_groups[0]['momentum'])



test_dataset = get_test_loader(config.predictdata_dir,config.batch_size,config.input_size,config.num_workers)

# losses = AverageMeter()
accs = AverageMeter()
#
# loss_fn = nn.CrossEntropyLoss()
#
#
# # print(model)
# # load the best checkpoint
model.to(device1)
model.eval()
#
# model1.cuda()
# model1.eval()
#
#
p = []
l = []
for i, (images, labels) in enumerate(test_dataset):

    images, labels = images.to(device1), labels.to(device1)
    images, labels = Variable(images), Variable(labels)
    cpu_label = labels.cpu().detach().numpy()
    for e in(cpu_label):
        l.append(e)

    outputs = model(images)

    prec = accuracy(outputs, labels)
    accs.update(prec, images.size()[0])

    out = F.log_softmax(outputs)
    score = out.cpu().detach().numpy()
    for e in(score):
        p.append(e[1])

print(accs.avg)
fpr, tpr, _ = roc_curve(l, p)
roc_auc = auc(fpr, tpr)
#
#
#
# p1 = []
# l1 = []
# for i, (images, labels) in enumerate(test_dataset):
#
#     images, labels = images.cuda(), labels.cuda()
#     images, labels = Variable(images), Variable(labels)
#     cpu_label = labels.cpu().detach().numpy()
#     for e in(cpu_label):
#         l1.append(e)
#
#     outputs = model1(images)
#
#     out = F.softmax(outputs)
#     score = out.cpu().detach().numpy()
#     for e in(score):
#         p1.append(e[1])
#
# fpr1, tpr1, _1 = roc_curve(l1, p1)
# roc_auc1 = auc(fpr1, tpr1)
#
#
#
plt.figure()
lw = 1
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %f)' % roc_auc)

# plt.plot(fpr1, tpr1, color='red',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc1)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


#     preds = outputs.argmax(dim=1)
#     loss = loss_fn(outputs, labels)
#
#     prec = accuracy(outputs, labels)
#     losses.update(loss.item(), images.size()[0])
#     accs.update(prec, images.size()[0])
#
# print(
#     '[*] Test loss: {:.3f}, acc: {:.6f}'.format(
#         losses.avg, accs.avg)
# )



# def fig_roc (model_num,model_path,test_dataset):
#     models = []
#     Y_true = []
#     Y_score = []
#     fpr = []
#     tpr = []
#     thresholds = []
#     roc_auc = []
#     colors = ['darkorange','red','darkgreen','tomato','peru']
#     for i in range(model_num):
#         m,o,e = loader_model(model_path[i])
#         models.append(m)
#         y_true = []
#         y_score = []
#         Y_true.append(y_true)
#         Y_score.append(y_score)
#
#     for j in range(model_num):
#         models[j].cuda()
#         models[j].eval()
#
#         for images, labels in test_dataset:
#             images, labels = images.cuda(), labels.cuda()
#             images, labels = Variable(images), Variable(labels)
#             cpu_label = labels.cpu().detach().numpy()
#             for l in (cpu_label):
#                 Y_true[j].append(l)
#
#             outputs = models[j](images)
#             out = F.softmax(outputs)
#             score = out.cpu().detach().numpy()
#             for s in (score):
#                 Y_score[j].append(s[1])
#
#         fpr_, tpr_, thresholds_ = roc_curve(Y_true[j], Y_score[j])
#         roc_auc_ = auc(fpr_, tpr_)
#         fpr.append(fpr_)
#         tpr.append(tpr_)
#         thresholds.append(thresholds_)
#         roc_auc.append(roc_auc_)
#
#     plt.figure()
#     lw = model_num
#     for k in range(model_num):
#         plt.plot(fpr[k], tpr[k], color=colors[k],
#                  lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[k])
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([-0.01, 1.0])
#     plt.ylim([0.0, 1.01])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     # plt.title('Receiver operating characteristic example')
#     plt.legend(loc="lower right")
#     plt.show()
#
#     return roc_auc
#
# auc = fig_roc(2,paths,test_dataset)
#
# for a in auc:
#     print(a)