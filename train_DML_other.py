import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import models

import os
import time
import shutil

from tqdm import tqdm
from utils import accuracy, AverageMeter, getWorkBook_DML
from tensorboard_logger import configure, log_value



class Trainer(object):


    def __init__(self, config, data_loader):

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
        self.ckpt_dir = '/media/cvnlp/3b670053-8188-42b6-a0aa-7390926a3303/home/cvnlp/LiChuanxiu/实验/DML/multi/40'
        self.logs_dir = config.logs_dir
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.resume = config.resume

        self.feature_extract = config.feature_extract
        self.use_pretrained = config.use_pretrained
        self.paramseed = config.paramseed

        self.print_freq = config.print_freq
        self.model_name = config.save_name
        self.model_num = config.model_num
        self.models = []
        self.optimizers = []
        self.schedulers = []
        self.mywb = getWorkBook_DML()


        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.loss_ce = nn.CrossEntropyLoss()
        self.best_valid_accs = [0.] * self.model_num

        self.device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



        for i in range(self.model_num):
            # build models
            if i == 0:
                model = models.get_googlenet(self.num_classes, self.feature_extract, self.use_pretrained, self.paramseed)
            if i == 1:
                model = models.inception_v3(self.num_classes, self.feature_extract, self.use_pretrained, self.paramseed)
            # model.cuda()
            # if torch.cuda.device_count() > 1:
            #     model = nn.s(model)
            # model.to(self.device)
            # if self.use_gpu:
            #     model.cuda()
            if self.use_gpu:
                model.to(self.device0)

            self.models.append(model)

            # initialize optimizer and scheduler
            optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum,
                                       weight_decay=self.weight_decay)

            self.optimizers.append(optimizer)

            # set learning rate decay
            scheduler = optim.lr_scheduler.StepLR(self.optimizers[i], step_size=self.step_size, gamma=self.gamma, last_epoch=-1)
            self.schedulers.append(scheduler)

        print('[*] Number of parameters of one model: {:,}'.format(
            sum([p.data.nelement() for p in self.models[0].parameters()])))

    def train(self):


        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
        )

        for epoch in range(self.epochs):

            for scheduler in self.schedulers:
                scheduler.step(epoch)

            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch + 1, self.epochs, self.optimizers[0].param_groups[0]['lr'], )
            )

            # train for 1 epoch
            train_losses, train_accs = self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_losses, valid_accs = self.validate(epoch)

            for i in range(self.model_num):
                is_best = valid_accs[i].avg > self.best_valid_accs[i]
                msg1 = "model_{:d}: train loss: {:.3f} - train acc: {:.3f} "
                msg2 = "- val loss: {:.3f} - val acc: {:.3f}"
                if is_best:
                    # self.counter = 0
                    msg2 += " [*]"
                msg = msg1 + msg2
                print(msg.format(i + 1, train_losses[i].avg, train_accs[i].avg, valid_losses[i].avg, valid_accs[i].avg))

                # check for improvement
                # if not is_best:
                # self.counter += 1
                # if self.counter > self.train_patience:
                # print("[!] No improvement in a while, stopping training.")
                # return
                self.best_valid_accs[i] = max(valid_accs[i].avg, self.best_valid_accs[i])
                self.save_checkpoint(i,
                                     {'epoch': epoch + 1,
                                      'model_state': self.models[i].state_dict(),
                                      'optim_state': self.optimizers[i].state_dict(),
                                      'best_valid_acc': self.best_valid_accs[i],
                                      }, is_best
                                     )
            self.record_loss_acc(train_losses, train_accs, valid_losses, valid_accs)
            dir = "/home/cvnlp/multi_googlenet_DML_inceptionv3_40X.xlsx"
            self.mywb.save(dir)

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = []
        accs = []

        for i in range(self.model_num):
            self.models[i].train()
            losses.append(AverageMeter())
            accs.append(AverageMeter())

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (images, labels) in enumerate(self.train_loader):
                if self.use_gpu:
                    images, labels = images.to(self.device0), labels.to(self.device0)
                    # images, labels = images.cuda(), labels.cuda()
                images, labels = Variable(images), Variable(labels)

                # forward pass
                outputs = []
                for model in self.models:
                    outputs.append(model(images))
                for i in range(self.model_num):
                    ce_loss = self.loss_ce(outputs[i], labels)
                    kl_loss = 0
                    for j in range(self.model_num):
                        if i != j:
                            kl_loss += self.loss_kl(F.log_softmax(outputs[i], dim=1),
                                                    F.softmax(Variable(outputs[j]), dim=1))
                    loss = ce_loss + kl_loss / (self.model_num - 1)

                    # measure accuracy and record loss
                    prec = accuracy(outputs[i], labels)
                    losses[i].update(loss.item(), images.size()[0])
                    accs[i].update(prec, images.size()[0])



                    # compute gradients and update SGD
                    self.optimizers[i].zero_grad()
                    loss.backward()
                    self.optimizers[i].step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "{:.1f}s - model1_loss: {:.3f} - model1_acc: {:.3f}".format(
                            (toc - tic), losses[0].avg, accs[0].avg
                        )
                    )
                )
                self.batch_size = images.shape[0]

                pbar.update(self.batch_size)

            return losses, accs

    def validate(self, epoch):

        losses = []
        accs = []
        for i in range(self.model_num):

            self.models[i].eval()
            losses.append(AverageMeter())
            accs.append(AverageMeter())

        for i, (images, labels) in enumerate(self.valid_loader):
            if self.use_gpu:
                images, labels = images.to(self.device0), labels.to(self.device0)
                # images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)

            # forward pass
            outputs = []
            for model in self.models:
                outputs.append(model(images))
            for i in range(self.model_num):
                ce_loss = self.loss_ce(outputs[i], labels)
                kl_loss = 0
                for j in range(self.model_num):
                    if i != j:
                        kl_loss += self.loss_kl(F.log_softmax(outputs[i], dim=1),
                                                F.softmax(Variable(outputs[j]), dim=1))
                loss = ce_loss + kl_loss / (self.model_num - 1)

                # measure accuracy and record loss
                prec = accuracy(outputs[i], labels)
                losses[i].update(loss.item(), images.size()[0])
                accs[i].update(prec, images.size()[0])


        return losses, accs



    def save_checkpoint(self, i, state, is_best):


        filename ='multi_googlenet_inceptionv3_NO'+ str(i + 1) + '_40X_ckpt_DML.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename ='multi_googlenet_inceptionv3_NO'+ str(i + 1) + '_40X_model_best_DML.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )

    def record_loss_acc(self,trainloss, trainacc, testloss, testacc):
        self.mywb["model1_trainloss"].append([trainloss[0].avg])
        self.mywb["model1_trainacc"].append([trainacc[0].avg])
        self.mywb["model1_testloss"].append([testloss[0].avg])
        self.mywb["model1_testacc"].append([testacc[0].avg])

        self.mywb["model2_trainloss"].append([trainloss[1].avg])
        self.mywb["model2_trainacc"].append([trainacc[1].avg])
        self.mywb["model2_testloss"].append([testloss[1].avg])
        self.mywb["model2_testacc"].append([testacc[1].avg])