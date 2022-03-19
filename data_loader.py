# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:12:10 2019

@author: chxy
"""

import numpy as np

import torch
from torchvision import datasets
from torchvision import transforms

def get_train_loader(data_dir,
                     batch_size,
                     random_seed,
                     input_size,
                     shuffle=True,
                     num_workers=0):
    """
    Utility function for loading and returning a multi-process
    train iterator over the CIFAR100 dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: train set iterator.
    """

    # define transforms
    trans = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize([0.78619444, 0.62567663, 0.76359934], [0.12863101, 0.1789347, 0.11344098])
    ])

    # load dataset
    dataset = datasets.ImageFolder(data_dir,trans)
    print('train_dataset')
    print(dataset)

    if shuffle:
        np.random.seed(random_seed)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print('train_loader')
    print(train_loader)

    return train_loader



def get_test_loader(data_dir,
                     batch_size,
                     input_size,
                     num_workers=0):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR100 dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    # define transforms
    trans = transforms.Compose([
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.78619444, 0.62567663, 0.76359934], [0.12863101, 0.1789347, 0.11344098])
    ])

    # load dataset
    dataset = datasets.ImageFolder(data_dir,trans)
    print('test_data')
    print(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers)
    print(data_loader)
    print(data_loader)

    return data_loader