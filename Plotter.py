
import numpy as np
import scipy.constants as scc
import scipy.special as special

import matplotlib.pyplot as plt

import os
from matplotlib.backends.backend_pdf import PdfPages       #For saving figures to single pdf

import scipy.signal as sig
import scipy.io as sio

import time
import timeit

''' global plotting settings '''
#plt.style.use('seaborn-paper')
# Update the matplotlib configuration parameters:
plt.rcParams.update({'text.usetex': False,
                     'lines.linewidth': 3,
                     'font.family': 'sans-serif',
                     'font.serif': 'Helvetica',
                     'font.size': 16,
                     'xtick.labelsize': 'large',
                     'ytick.labelsize': 'large',
                     'axes.labelsize': 'large',
                     'axes.titlesize': 'large',
                     'axes.grid': True,
                     'grid.alpha': 0.53,
                     'lines.markersize': 12,
                     'legend.borderpad': 0.2,
                     'legend.fancybox': True,
                     'legend.fontsize': 'medium',
                     'legend.framealpha': 0.7,
                     'legend.handletextpad': 0.1,
                     'legend.labelspacing': 0.2,
                     'legend.loc': 'best',
                     'figure.figsize': (12,8),
                     'savefig.dpi': 100,
                     'pdf.compression': 9})


plot_loc = "plots/{}"  # save plots in the "plots" folder
data_loc = "data/{}"   # save data in "data" folder


""" load data from data_loc based on expected name """
CNN_train_accs = np.load(data_loc.format("CNN_train_accs.npy"))
CNN_val_accs = np.load(data_loc.format("CNN_val_accs.npy"))

# create batch size x-axis for CNN (list of nums from 1-20)
epochs = [x + 1 for x in range(20)]

Resnet_val_accs = np.load("data/Resnet_val_accs.npy")
Resnet_val_accs_xaxis = np.load("data/Resnet_val_accs_xaxis.npy")
Resnet_train_accs = np.load("data/Resnet_train_accs.npy")
Resnet_train_accs_xaxis = np.load("data/Resnet_train_accs_xaxis.npy")

""" plotting """
fig = plt.figure()
plt.plot(epochs, CNN_train_accs, label="CNN Training Acc", color="b", linestyle="--")
plt.plot(epochs, CNN_val_accs, label="CNN Validation Acc", color="b")

plt.plot(Resnet_train_accs_xaxis, Resnet_train_accs, label="Resnet Training Acc", color="r", linestyle="--")
plt.plot(Resnet_val_accs_xaxis, Resnet_val_accs, label="Resnet Val_Acc", color="r")

# need to do this to add a 0 tick mark
x_ticks_epochs = epochs.insert(0,0)
plt.xticks(x_ticks_epochs)

plt.xlabel("Epoch No.")
plt.ylabel("Accuracy [%]")

plt.title("Resnet Vs CNN Comparison: Identifying American Sign Language Handsigns")

plt.legend()
plt.show()
