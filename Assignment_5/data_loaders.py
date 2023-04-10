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
                     train_transf=None,
                     test_transf=None,
                     img_shape=None,
                     use_cutmix=False,
                     ):
    """
    Loads the training data for the specified dataset.
    :param dataset_name: Name of the dataset to load.
    :param batch_size: Batch size to use for training.
    :param shuffle_training_data: Whether to shuffle the training data.
    :param transform: Transform to apply to the training data.
    :return: Training data loader.
    

    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])
    test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])
   

    if train_transf is not None:
        train_transform = train_transf
    if test_transf is not None:
        test_transform = test_transf

    if use_cutmix:
        collator = CutMixCollator(alpha=.001)
    else:
        collator = torch.utils.data.dataloader.default_collate


    # add resize transform if img_shape is not None
    if img_shape is not None:
        train_transform = transforms.Compose([
                                train_transform,
                                transforms.Resize(img_shape)
        ])
        test_transform = transforms.Compose([
                                test_transform,
                                transforms.Resize(img_shape)
        ])
    

    train_dataset_aug = None
        

    if(dataset_name == 'cifar10'):
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    elif(dataset_name == 'mnist'):
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    elif(dataset_name == 'fashionmnist'):
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)

    elif(dataset_name == 'stanfordcars'):
        # train_dataset = StanfordCars(root='./data', split="train", download=True, transform=train_transform)
        # test_dataset = StanfordCars(root='./data', split="test", download=True, transform=test_transform)
        train_dataset = datasets.StanfordCars(root="./data", split="train", transform=train_transform, download=True)
        test_dataset = datasets.StanfordCars(root="./data", split="test", transform=test_transform, download=True)

    elif(dataset_name == 'svhn'):
        # train_dataset = StanfordCars(root='./data', split="train", download=True, transform=train_transform)
        # test_dataset = StanfordCars(root='./data', split="test", download=True, transform=test_transform)
        train_dataset = datasets.SVHN(root="./data", split="train", transform=train_transform, download=True)
        test_dataset = datasets.SVHN(root="./data", split="test", transform=test_transform, download=True)
        


    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=shuffle_training_data,
                                           collate_fn=collator,
                                           num_workers=4,
                                           drop_last=True
                                           ) 

    eval_loader =  torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            num_workers=4,
                                            shuffle=False)
    return train_loader, eval_loader


class CutMixCriterion:
    def __init__(self, reduction):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(self, preds, targets):
        targets1, targets2, lam = targets
        return lam * self.criterion(
            preds, targets1) + (1 - lam) * self.criterion(preds, targets2)

class CutMixCollator:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = cutmix(batch, self.alpha)
        return batch


def cutmix(batch, alpha):
    data, targets = batch

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    # Draw samples from a Beta distribution. The Beta distribution is a special case of the Dirichlet distribution,
    # and is related to the Gamma distribution.
    #lam = np.random.beta(alpha, alpha)/5
    lam = np.random.uniform(0.5, 0.9)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets = (targets, shuffled_targets, lam)

    return data, targets

