#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:04:08 2020

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:38:21 2020

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from PIL import Image

seed = 7
np.random.seed(seed)
torch.manual_seed(seed)

# Designating GPU usage
device = torch.device("cuda:0")

IMG_PATH = "/Users/alex/Documents/Python/asl-alphabet/asl_alphabet_train/"

# Functions for loading images, shuffling data, and calculating accuracy of network predictions
def load_images(letter, N = 10):
    arrays = []
    for i in range(N):
        image = Image.open(IMG_PATH+letter+"/"+letter+str(i+1)+".jpg")
        array = np.asarray(image)
        arrays.append(array)
    return np.array(arrays)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def accuracy(est_labels, labels):
    max_vals = torch.argmax(est_labels, 1)
    acc = (max_vals == labels).float().mean()
    return acc

# Letter lookup dict for easy indexing
letter_lookup = {letter: i for i, letter in enumerate(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
                                                       "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"])}

# Loads and shuffles training data in a pairwise manner
train_num = 20 # the number of each letter to load for training (out of a total of 3000)
train_in = []
train_out = []
for letter in letter_lookup.keys():
    arrays = load_images(letter,train_num)
    for array in arrays:
        train_in.append(array)
        train_out.append(letter_lookup[letter])
      
train_in, train_out = unison_shuffled_copies(np.array(train_in), np.array(train_out))

# Loads and shuffles testing data in a pairwise manner
test_num = 20 # the number of each letter to load for training (out of a total of 3000)
test_in = []
test_out = []
for letter in letter_lookup.keys():
    arrays = load_images(letter,test_num)
    for array in arrays:
        test_in.append(array)
        test_out.append(letter_lookup[letter])
      
test_in, test_out = unison_shuffled_copies(np.array(test_in), np.array(test_out))

# Rearranging image dimensions to be compatible with PyTorch
train_in = np.moveaxis(train_in, -1, 1)
test_in = np.moveaxis(test_in, -1, 1)

# Normalizing data
train_in = train_in / 255.
test_in = test_in / 255.

# Ensuring type compatibility 
train_in = torch.from_numpy(np.float32(train_in))
train_out = torch.from_numpy(train_out).long()
test_in = torch.from_numpy(np.float32(test_in))
test_out = torch.from_numpy(test_out).long()

# Network hyperparameters
learn_rate = .005
epochs = 10
b_frac = .1
batches = int(1/b_frac)
b_size = int(b_frac*train_in.shape[0])

# Defines a CNN class inheriting from the Module base class
class Sign_Net(nn.Module):
    def __init__(self):
        super(Sign_Net, self).__init__()
        self.c1 = nn.Conv2d(3, 6, 5)
        self.c2 = nn.Conv2d(6, 16, 5)
        self.c_drop = nn.Dropout2d(.4)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16 * 47**2, 100)
        self.fc2 = nn.Linear(100, 29)
        
    def forward(self, x):
        # Pooled relu activated output from first convolutional layer
        x = self.pool(F.relu(self.c1(x)))
        # Dropout of 40% on second convolutional layer
        x = self.c_drop(self.c2(x))
        # Pooled relu activated output from second convolutional layer
        x = self.pool(F.relu(x))
        # Determining input dim for first fully connected layer
        x = x.view(b_size, 16 * 47**2)
        # Relu activated output from first fully connected layer
        x = F.relu(self.fc1(x))
        # Activationless output from second fully connected layer
        x = self.fc2(x)
        return x

# Instantiating the network and defining the optimizer
net = Sign_Net().to(device)
loss = nn.CrossEntropyLoss()
opti = opt.Adam(net.parameters(), lr = learn_rate)

# Data iteration loop for training
for e in range(epochs):
    for b in range(batches):
        b_start = b * b_size
        b_end = (b+1) * b_size
        batch_in = train_in[b_start : b_end].to(device)
        batch_out = train_out[b_start : b_end].to(device)

        # Zeroes out gradient parameters 
        opti.zero_grad() 
        
        # Predicted output as determined by current network state
        train_pred = net(batch_in)
        
        # Computes loss and accuracy of network predictions with respect to actual labels
        train_loss = loss(train_pred, batch_out)
        train_acc = accuracy(train_pred, batch_out)
        
        # Back propagation of gradient
        train_loss.backward()
        
        # Adjusts weights
        opti.step()
        
    print("Epoch: " + str(e+1) + ", Accuracy: " + str(round(train_acc.item(),2)))

# Testing iteration loop
test_accs = []
# Batch size recalculated for test data
b_size = int(b_frac*test_in.shape[0])
for b in range(batches):
        b_start = b * b_size
        b_end = (b+1) * b_size
        batch_in = test_in[b_start : b_end].to(device)
        batch_out = test_out[b_start : b_end].to(device)  
        
        # Computes loss and accuracy of network predictions with respect to actual labels
        test_pred = net(batch_in)
        test_accs.append(accuracy(test_pred, batch_out).item())

print("Testing accuracy: " + str(round(sum(test_accs)/batches,2)))


