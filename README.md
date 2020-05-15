# 476Resnet
Machine Learning 476 @ UMDCP, Cameron/Alex/Angel/Jorge Final Project


# How to Run this Code

(1) clone this repo into its own folder

(2) create an empty directory called "data" so the .py files have somewhere to store their saved .npy

(3) download the dataset from Kaggle at [this link](https://www.kaggle.com/grassknoted/asl-alphabet).

- (A) download the .zip file for the dataset, and unzip it into the directory, making sure the images are only one folder deep (e.g. `/asl-alphabet_train/A/A1.jpg`, instead of `asl-alphabet_train/asl-alphabet_train/A/A1.jpg`) which will happen when unzipping
   
- (B) if only using a terminal, use the kaggle API: `pip3 install kaggle`, `kaggle datasets download -d grassknoted/asl-alphabet`, then `mv` and `unzip` the file, and finally use `rsync` to fix the images being two folders deep
   
(4) run the python scripts with `python3 <scriptname>.py`

(5) run the plotting script `python3 Plotter.py`


# Intro
We are comparing two strategies for image classification
the first is using ResNet-50 as a feature extractor to pre-process the data and feed the output into our densely connected network, which will be presented by Cameron
and a convolutional neural net to directly classify the images, which will be presented by Alex

Feature Extraction is dimensionality reduction.
Consistent shapes within the dataset are determined as feature vectors which will be directly trained. For ResNet, which is one of the strategies we are using, a 2048-dimensional vector is created to more precisely distinguish key features in images.
Multiple people are able to use the same feature extractor due to low level features already being extracted, so higher order features can be trained in individual neural networks

In this class, the convolutional neural nets we've created in the past perform low-level extraction early on, as Alex will discuss. If within the second example, we had the last convolutional layer feed into a different fully-connected neural net, this would mimick what we acheive through ResNet.
ResNet however, was already extensively trained through multiple different datasets, and as such can be used in various applications, while creating a similar feature extractor through our code would require more rigorous training than just the dataset we are using.
Once the pre-processing through a feature extractor is completed, network specific feature vectors can be trained and hyperparameters can be tuned.


To name a few further applications of this process, it can be applied to medical imaging for detecting tumors and abnormalities through scans, classifying huge documents and texts in databases by genre or keyword, or removing unusual objects in astronomical imaging from telescope captures
