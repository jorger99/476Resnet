#!/usr/bin/env python
# coding: utf-8



# General imports
import keras.applications
from keras import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np



# Change these to the directories that you want to keep stuff in accordingly
IMG_PATH = "asl-alphabet_train/"
ARRAY_PATH = "data/"



# Initialize a keras Resnet50 model (with pretrained weights) with the right settings to accept our images
ResNet50Preprocessor = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(200,200,3),
    pooling='avg',

)



# Define some functions to make stuff smoother later on
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



letter_lookup = {letter: i for i, letter in enumerate(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
                                                       "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"])}
number = 20 # the number of each letter to load (out of a total of 3000)
letter_range = 0, 1



# This is a moderately long process which converts the images into respective 2048 vectors and saves them in a similar format to the images
for letter in letter_lookup.keys():
    print(letter, end=' ')
    np.save( ARRAY_PATH+"/"+str(number)+letter+".npy", ResNet50Preprocessor.predict(load_images(letter,number)))
print("\nFinished!")


# This loads the Letter-wise 2048 vectors into a single input and output vector pair then shuffles them (and saves them for good measure)
Xs = []
Ys = []
Xvals = []
Yvals = []
manual_val_split = .7
if manual_val_split != None: split_index = int(manual_val_split*number)
for letter in letter_lookup.keys():
    arrays = np.load(ARRAY_PATH+"/"+str(number)+letter+".npy")
    X = []
    Y = []
    for array in arrays:
        X.append(array)
        Y.append(letter_lookup[letter])

    if manual_val_split != None:
        Xvals += X[:split_index]
        Yvals += Y[:split_index]
        Xs += X[split_index:]
        Ys += Y[split_index:]
    else:
        Xs += X[:]
        Ys += Y[:]

Xs, Ys = unison_shuffled_copies(np.array(Xs), np.array(Ys))
#np.save(ARRAY_PATH+'inputs.npy', Xs)
#np.save(ARRAY_PATH+'labels.npy', Ys)
print('Training samples:',Ys.shape[0])

if manual_val_split != None:
    Xvals, Yvals = unison_shuffled_copies(np.array(Xvals), np.array(Yvals))
    #np.save(ARRAY_PATH+'inputsval.npy', Xvals)
    #np.save(ARRAY_PATH+'labelsval.npy', Yvals)
    print('Testing  samples:',Yvals.shape[0])


optimizer = optimizers.Adam(learning_rate=0.004)

model = Sequential()
model.add(Dropout(0.0))
model.add(Dense(29, activation = 'softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])


history = model.fit(
    Xs,
    Ys,
    #validation_split=.8,                # use this when not using manual_val_split
    validation_data = [Xvals, Yvals],    # use this when using manual_val_split
    verbose = 0,
    epochs =  10,
    batch_size=32)
plt.plot(np.arange(.5,len(history.history['loss'])+.5,1),history.history['accuracy'])
#plt.show()
plt.plot(np.arange(1,len(history.history['loss'])+1,1),history.history['val_accuracy'])
plt.show()
print('Final val_accuracy:', history.history['val_accuracy'][-1])
