# 476Resnet
Machine Learning 476 @ UMDCP, Cameron/Alex/Angel/Jorge Final Project

We are comparing two strategies for image classification
Using ResNet-50 as a feature extractor to pre-process the data and feed the output into our densely connected network, which will be presented by Cameron
and a convolutional neural net to directly classify the images, which will be presented by Alex

Feature Extraction is dimensionality reduction.
Consistent shapes within the dataset are determined as feature vectors which will be directly trained. For ResNet, one of the strategies we are using, a 2048-dimensional feature vector is created to more precisely distinguish images.
Multiple people are able to use the same feature extractor due to low level features already being extracted, so higher order features can be trained in individual neural networks

In this class, the convolutional neural nets we've created in the past perform low-level extraction early on, as Alex will discuss. If within the second example, we had a the last convolutional layer feed into a different fully-connected neural net, this would mimick the intention of ResNet
ResNet however, was already extensively trained, and as such can be used in various applications, while creating a similar feature extractor through our code would require more rigorous training than just the dataset we are using.
Once the pre-processing through a feature extractor is completed, network specific feature vectors can be trained and hyperparameters can be tuned.
