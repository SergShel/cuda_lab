''' models.py '''

import torch 
import torch.nn as nn
import sklearn.metrics
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import numpy as np



class CIFAR10_MLP_Classifier(torch.nn.Module):

    def __init__(self, num_neurons=[50, 20], activation=torch.nn.ReLU):
        super().__init__()
        num_neurons = [32*32*3] + num_neurons

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Flatten())
        for i, in_neurons in enumerate(num_neurons[:-1]):
            out_neurons = num_neurons[i+1]
            self.layers.append(torch.nn.Linear(in_neurons, out_neurons, True))
            self.layers.append(activation())
        self.layers.append(torch.nn.Linear(num_neurons[-1], 10, True))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CIFAR10_CNN_Classifier(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 32 x 16 x 16

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 8 x 8

            nn.Flatten(), 
            nn.Linear(256*4*4, 100),
            nn.ReLU(),
            nn.Linear(100, 10))
        
    def forward(self, xb):
        return self.network(xb.cuda())


