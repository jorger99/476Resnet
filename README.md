# 476Resnet
Machine Learning 476 @ UMDCP, Cameron/Alex/Angel/Jorge Final Project

We are comparing two strategies for image classification
Using ResNet-50 as a feature extractor to pre-process the data and feed the output into our densely connected network, which will be presented by Cameron
and a convolutional neural net to directly classify the images, which will be presented by Alex

Feature Extraction is dimensionality reduction.
Consistent shapes within the dataset are determined as feature vectors which will be directly trained. For ResNet, one of the strategies we are using, a 2048-dimensional feature vector is created to more precisely distinguish images.
Multiple people are able to use the same feature extractor due to low level features already being extracted, so higher order features can be trained in individual neural networks
