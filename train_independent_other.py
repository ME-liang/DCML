import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from utils import accuracy, AverageMeter, getWorkBook
from VGG16 import *

import os
import time
import shutil

from tqdm import tqdm
#from utils import accuracy, AverageMeter
import models
from tensorboard_logger import configure, log_value


class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for
    training the MobileNet Model.
 bmnmnbnmmn;'
    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        self.config = config

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.dataset)
            self.num_valid = len(self.valid_loader.dataset)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)

        self.num_classes = config.num_classes

        # training params
        self.epochs = config.epochs
        self.momentum = config.momentum
        self.lr = config.init_lr
        self.weight_decay = config.weight_decay
        self.gamma = config.gamma
        self.step_size = config.step_size

        # misc params
        self.use_gpu = config.use_gpu
        self.feature_extract = config.feature_extract
        self.use_pretrained = config.use_pretrained
        self.paramseed = config.paramseed
        self.loss_ce = nn.CrossEntropyLoss()
        self.best_valid_accs = 0.
        self.model_name = config.save_name
        self.ckpt_dir = '/media/cvnlp/3b670053-8188-42b6-a0aa-7390926a3303/home/cvnlp/LiChuanxiu/实验/resnet50/multi/400'
        self.logs_dir = config.logs_dir
        self.mywb = getWorkBook()

        self.device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


        self.model = models.get_resnet50(self.num_classes, False, False, 0)



        if self.use_gpu:
            self.model.to(self.device1)

        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)
        # self.model.cuda()

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                              weight_decay=self.weight_decay)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma, last_epoch=-1)

        print('[*] Number of parameters of one model: {:,}'.format(

            sum([p.data.nelement() for p in self.model.parameters()])))

    def train(self):


        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
        )

        for epoch in range(self.epochs):


            self.scheduler.step(epoch)

            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch + 1, self.epochs, self.optimizer.param_groups[0]['lr'], )
            )

            # train for 1 epoch
            train_losses, train_accs = self.train_one_epoch(epoch)
            # evaluate on validation set
            valid_losses, valid_accs = self.validate(epoch)


            is_best = valid_accs.avg > self.best_valid_accs
            msg1 = "model_: train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f}"
            if is_best:
                msg2 += " [*]"
            msg = msg1 + msg2
            print(msg.format(train_losses.avg, train_accs.avg, valid_losses.avg, valid_accs.avg))
            self.record_loss_acc(train_losses.avg, train_accs.avg, valid_losses.avg, valid_accs.avg)
            self.best_valid_accs = max(valid_accs.avg, self.best_valid_accs)
            self.save_checkpoint(epoch,
                                 {'epoch': epoch + 1,
                                  'model_state': self.model.state_dict(),
                                  'optim_state': self.optimizer.state_dict(),
                                  'best_valid_acc': self.best_valid_accs,
                                  }, is_best
                                 )
            dir = "/home/cvnlp/resnet50_multi_e50_lr001_400X.xlsx"
            self.mywb.save(dir)





    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        self.model.train()

        tic = time.time()


        with tqdm(total=self.num_train) as pbar:
            for i, (images, labels) in enumerate(self.train_loader):
                if self.use_gpu:
                    images, labels = images.to(self.device1), labels.to(self.device1)
                images, labels = Variable(images), Variable(labels)
                outputs = self.model(images)
                loss = self.loss_ce(outputs, labels)

                prec = accuracy(outputs, labels)
                losses.update(loss.item(), images.size()[0])
                accs.update(prec, images.size()[0])

                # compute gradients and update SGD
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "{:.1f}s - model1_loss: {} - model1_acc: {:.6f}".format(
                            (toc - tic), losses.avg, accs.avg
                        )
                    )
                )
                self.batch_size = images.shape[0]

                pbar.update(self.batch_size)

            return losses, accs


    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()
        self.model.eval()


        for i, (images, labels) in enumerate(self.valid_loader):
            if self.use_gpu:
                images, labels = images.to(self.device1), labels.to(self.device1)
            images, labels = Variable(images), Variable(labels)

            outputs = self.model(images)
            loss = self.loss_ce(outputs, labels)
            prec = accuracy(outputs, labels)
            losses.update(loss.item(), images.size()[0])
            accs.update(prec, images.size()[0])

        return losses, accs


    def save_checkpoint(self, i, state, is_best):
        filename = 'resnet50_multi_e50_lr001_400X_ckpt_'+str(i)+'.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = 'resnet50_multi_e50_lr001_400X_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )

    def record_loss_acc(self,epoch_trainloss, epoch_trainacc, epoch_testloss, epoch_testacc):
        self.mywb["epoch_trainloss"].append([epoch_trainloss])
        self.mywb["epoch_trainacc"].append([epoch_trainacc])
        self.mywb["epoch_testloss"].append([epoch_testloss])
        self.mywb["epoch_testacc"].append([epoch_testacc])