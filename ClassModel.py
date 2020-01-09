# from scipy.misc import imread
from imageio import imread

import math
import numpy as np
import cv2
import keras
import seaborn as sns
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential


class ClassModel:
    def __init__(self, conv_num, kernel_size, filter_number, dense_num, conv_dropout, dense_dropout, activation,
                 pool_size, hidden_num_units, output_layer_num, pooling=None):
        self.conv_num = conv_num
        self.kernel_size = kernel_size
        self.filter_number = filter_number
        self.dense_num = dense_num
        self.conv_dropout = conv_dropout
        self.dense_dropout = dense_dropout

        self.model = Sequential()

        for i in range(1, conv_num+1):
            # Conv2D(64, (3, 3), activation='relu', padding='same'),
            # BatchNormalization(),
            if i == 1:
                self.model.add(Conv2D(filter_number*2**i, kernel_size, activation=activation, input_shape=(64, 64, 3), padding='same'))
                self.model.add(BatchNormalization())
            else:
                self.model.add(Conv2D(filter_number*2**i, kernel_size, activation=activation, padding='same'))
                self.model.add(BatchNormalization())

            self.model.add(Conv2D(filter_number*2**i, kernel_size, activation=activation, padding='same'))
            self.model.add(BatchNormalization())

            if pooling == 'averagepooling':
                self.model.add(AveragePooling2D(pool_size=pool_size))
            else:
                self.model.add(MaxPooling2D(pool_size=pool_size))

            self.model.add(Dropout(conv_dropout))

        self.model.add(Flatten())

        flatten_out = self.model.layers[-1].output_shape[1]/2

        for i in range(1, dense_num):
            self.model.add(Dense(units=int(flatten_out/i), activation='relu'))
            self.model.add(Dropout(dense_dropout))

        self.model.add(Dense(units=output_layer_num * 30, activation='relu'))
        self.model.add(Dropout(dense_dropout))
        self.model.add(Dense(units=output_layer_num, input_dim=flatten_out, activation='softmax'))
        # self.model.add(Dense(units=output_layer_num*30, activation='softmax'))
        # self.model.add(Dense(units=output_layer_num, activation='softmax'))