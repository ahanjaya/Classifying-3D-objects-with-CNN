#!/usr/bin/env python3

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.models import load_model, Sequential
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv3D, MaxPooling3D, LeakyReLU
from keras.layers.core import Dense, Dropout, Activation, Flatten

class Model:
    def __init__(self):
        self.growth_memory() # limit tensorflow to use all of GPU resources
        self.number_class = 4

    def growth_memory(self):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for k in range(len(physical_devices)):
                tf.config.experimental.set_memory_growth(physical_devices[k], True)
                print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
        else:
            print("Not enough GPU hardware devices available")

    def voxnet(self):
        model = Sequential()
        model.add(Conv3D(32, input_shape=(32, 32, 32, 1), kernel_size=(5, 5, 5), strides=(2, 2, 2), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_last',))
        
        model.add(Flatten())
        model.add(Dense(128, activation='linear'))
        model.add(BatchNormalization())
        model.add(Dense(units=self.number_class, activation='softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        # print(model.metrics_names)

        return model

    def MV_CNN_1(self):
        model = Sequential()
        model.add(Conv3D(32, input_shape=(32, 32, 32, 1), kernel_size=(3, 3, 3), strides=(1, 1, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_last',))
        
        model.add(Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_last',))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(2048, activation='linear'))
        model.add(BatchNormalization())
        model.add(Dense(units=self.number_class, activation='softmax'))
        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        # print(model.metrics_names)

        return model

    def run(self):
        # model_voxnet   = self.voxnet()
        model_mvcn1    = self.MV_CNN_1()
       
if __name__ == '__main__':
    model = Model()
    model.run()