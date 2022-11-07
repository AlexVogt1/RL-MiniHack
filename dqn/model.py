#imports
import torch
import torch.nn.functional as F
import torch.nn as nn
from gym import spaces
from torch.optim import Optimizer
import numpy as np

class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network.
    """

    #inti the DQN that takes in an the action space as a param so that the output is action list probabilities
    def __init__(self, action_space: spaces.Discrete):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20,kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50,kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = nn.Linear(in_features=1600, out_features=500)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=500, out_features=action_space.n)

    #forward pass for network
    def forward(self, x):
        # define first conv layer with max pooling
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # define second conv layer with max pooling
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # Define fully connected layers
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x