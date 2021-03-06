from torchvision import models
import torch.nn as nn
from senet.se_resnet1 import se_resnet50
from senet.newsext101 import se_resnext101_32x4d
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os, glob
import scipy.io as sio
import torch.hub


def set_parameter_requires_grad(model, feature_extract, paramseed):
    if feature_extract:
        for index,param in enumerate(model.parameters()):
            if paramseed==0:
                param.requires_grad = False
            elif index%paramseed==0:
                param.requires_grad = True
            else:
                param.requires_grad = False

def qoery_param(model):
    for index, param in enumerate(model.parameters()):
        print(param.requires_grad)

def get_resnet34(num_classes, feature_extract, use_pretrained, paramseed):
    model_ft = models.resnet34(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract,paramseed)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft

def get_resnet50(num_classes, feature_extract, use_pretrained, paramseed):
    model_ft = models.resnet50(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract,paramseed)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft

def get_se_resnet50(num_classes, feature_extract, use_pretrained, paramseed):
    model_ft = se_resnet50(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract, paramseed)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft

def get_se_resnetxt101(num_classes, feature_extract, use_pretrained, paramseed):
    if use_pretrained == True:
        CC = 'imagenet'
        model_ft = se_resnext101_32x4d(pretrained=CC)
    else:
	    model_ft = se_resnext101_32x4d()
    set_parameter_requires_grad(model_ft, feature_extract, paramseed)
    num_ftrs = model_ft.last_linear.in_features
    model_ft.last_linear = nn.Linear(num_ftrs, num_classes)
    return model_ft

def get_googlenet(num_classes, feature_extract, use_pretrained, paramseed):
    model_ft = models.googlenet(pretrained=use_pretrained)
    model_ft.aux_logits = False
    set_parameter_requires_grad(model_ft, feature_extract,paramseed)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,num_classes)

    return model_ft

def get_vgg16(num_classes, feature_extract, use_pretrained, paramseed):
    model_ft = models.vgg16(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract, paramseed)
    sequential = model_ft.classifier
    num_ftrs = sequential[6].in_features
    sequential[6] = nn.Linear(num_ftrs, num_classes)
    return model_ft

def get_alexnet(num_classes, feature_extract, use_pretrained, paramseed):
    model_ft = models.alexnet(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract, paramseed)
    sequential = model_ft.classifier
    num_ftrs = sequential[6].in_features
    sequential[6] = nn.Linear(num_ftrs, num_classes)
    return model_ft

def inception_v3(num_classes, feature_extract, use_pretrained, paramseed):
    model_ft = models.inception_v3(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract, paramseed)

    model_ft.aux_logits = False

    num_ftrs = model_ft.AuxLogits.fc.in_features
    model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    # Handle the primary net
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft

def densenet(num_classes, feature_extract, use_pretrained, paramseed):
    model_ft = models.densenet121(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract, paramseed)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    return model_ft


