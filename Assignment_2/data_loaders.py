import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def load_train_data(dataset_name,
                     batch_size, 
                     shuffle_training_data=True, 
                     add_augmentation=False,
                     ):
    """
    Loads the training data for the specified dataset.
    :param dataset_name: Name of the dataset to load.
    :param batch_size: Batch size to use for training.
    :param shuffle_training_data: Whether to shuffle the training data.
    :param transform: Transform to apply to the training data.
    :return: Training data loader.
    

    """
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
    transform_aug = transforms.Compose([
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomCrop(32, padding=4),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])

    train_dataset_aug = None
        

    if(dataset_name == 'cifar10'):
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        if(add_augmentation):
            train_dataset_aug = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_aug)

    elif(dataset_name == 'mnist'):
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        if(add_augmentation):
            train_dataset_aug = datasets.MNIST(root='./data', train=True, download=True, transform=transform_aug)

    elif(dataset_name == 'fashionmnist'):
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        if(add_augmentation):
            train_dataset_aug = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_aug)

    if(add_augmentation):
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_aug])

    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=shuffle_training_data) 

    eval_loader =  torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
    return train_loader, eval_loader
