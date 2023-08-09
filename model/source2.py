"""
EECS 445 - Introduction to Machine Learning
Winter 2023 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config

class Source2(nn.Module):
    def __init__(self):
        """Define the architecture, i.e. what layers our network contains. 
        At the end of __init__() we call init_weights() to initialize all model parameters (weights and biases)
        in all layers to desired distributions."""
        super().__init__()

        ## TODO: define each layer of your network
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5,5), stride=(2,2))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(2,2), stride=(1,1))
        self.fcon1 = nn.Linear(in_features=32,out_features=64)
        self.fcon2 = nn.Linear(in_features=64, out_features=8)
        ##

        self.init_weights()

    def init_weights(self):
        """Initialize all model parameters (weights and biases) in all layers to desired distributions"""
        ## TODO: initialize the parameters for your network
        torch.manual_seed(42)
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        ## TODO: initialize the parameters for [self.fc1]
        Cin = self.fcon1.weight.size(1)
        nn.init.normal_(self.fcon1.weight, 0.0, 1 / sqrt(Cin))
        nn.init.constant_(self.fcon1.bias, 0.0)
        
        Cin = self.fcon2.weight.size(1)
        nn.init.normal_(self.fcon2.weight, 0.0, 1 / sqrt(Cin))
        nn.init.constant_(self.fcon2.bias, 0.0)
        ##

    def forward(self, x):
        """This function defines the forward propagation for a batch of input examples, by
            successively passing output of the previous layer as the input into the next layer (after applying
            activation functions), and returning the final output as a torch.Tensor object

            You may optionally use the x.shape variables below to resize/view the size of
            the input matrix at different points of the forward pass"""
        N, C, H, W = x.shape

        ## TODO: forward pass
        z = self.conv1(x)
        z = self.pool(F.relu(z))
        z = self.conv2(z)
        z = self.pool(F.relu(z))
        z = self.conv3(z)
        z = self.pool(F.relu(z))
        z = F.relu(self.conv4(z))
        z = torch.flatten(z, start_dim=1)
        z = F.relu(self.fcon1(z))
        z = self.fcon2(z)
        #z = F.softmax(self.fcon2(z),dim=1)
        ##

        return z
