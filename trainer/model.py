#!/usr/bin/env python3

"""Model to classify mugs

This file contains all the model information: the training steps, the batch
size and the model itself.
"""

BATCH_SIZE = 32
NUM_TRAINING_STEPS = 10 #20000


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import layers
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

def get_batch_size():
    """Returns the batch size that will be used by your solution.
    It is recommended to change this value.
    """
    return BATCH_SIZE

def get_epochs():
    """Returns number of epochs that will be used by your solution.
    It is recommended to change this value.
    """
    return NUM_TRAINING_STEPS

def solution(input_layer):
    """Returns a compiled model.

    This function is expected to return a model to identity the different mugs.
    The model's outputs are expected to be probabilities for the classes and
    and it should be ready for training.
    The input layer specifies    the shape of the images. The preprocessing
    applied to the images is specified in data.py.

    Add your solution below.

    Parameters:
        input_layer: A tf.keras.layers.InputLayer() specifying the shape of the input.
            RGB colored images, shape: (width, height, 3)
    Returns:
        model: A compiled model
    """

    # TODO: Code of your solution
        
    #model = Sequential()
    #model.add(Conv2D(filters=16, kernel_size=(4, 4), activation="relu", input_shape=(64, 64, 3)))
    
    #model.add(Flatten(input_shape=(64, 64, 3))) 
    #model.add(Dense(64, activation="relu")) 

    #model.add(Dropout(0.6))
    #model.add(Dense(32, activation=layers.LeakyReLU(alpha=0.3))) 
    #model.add(Dropout(0.5))
    #model.add(Dense(64, activation=layers.LeakyReLU(alpha=0.3))) 
    #model.add(Dropout(0.3))
    #model.add(Flatten())
    
    #model.add(Dense(4, activation='softmax'))

    """model = Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    """

    """model = Sequential([
      layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D((2, 2)),
      #layers.Conv2D(64, 3, padding='same', activation='relu'),
      #layers.MaxPooling2D((2, 2)),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(4, activation='softmax')
    ])"""

    """model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, c, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')])"""

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(4, activation='softmax'))


    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    # TODO: Return the compiled model
    return model































