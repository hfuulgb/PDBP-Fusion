# -*- coding: utf-8 -*-
import os
from keras import layers, models
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, SeparableConv1D
from keras.layers import LSTM, Bidirectional
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
from keras.optimizers import *
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import pdb
import utils.tools as utils
from datetime import datetime
from keras.applications import ResNet50


def get_cnn_network_1layer():
    # nbfilter = 141
    model = Sequential()
    model.add(
        Convolution1D(128,
                      3,
                      strides=1,
                      padding='valid',
                      input_shape=(140, 5),
                      activation="relu",
                      kernel_initializer='random_uniform',
                      name="convolution_1d_layer"))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.3))
    model.add(
        Convolution1D(128,
                      3,
                      strides=1,
                      padding='valid',
                      activation="relu",
                      kernel_initializer='random_uniform',
                      name="conv_1d_layer3"))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.25))
    # model.add(Dense(2, activation='sigmoid'))
    model.add(layers.Dense(2, activation='softmax'))
    return model


def get_cnn_network_2layer():
    inputs = layers.Input(shape=(1000, 20), name="input_6")
    x = Convolution1D(64,
                      7,
                      strides=1,
                      padding='valid',
                      kernel_initializer='he_normal',
                      activation="relu",
                      name="conv1d_11")(inputs)
    x = layers.MaxPooling1D(pool_size=3, name="max_pooling1d_11")(x)
    x = layers.Dropout(0.2, name="dropout_21")(x)

    x = Convolution1D(64,
                      7,
                      strides=1,
                      padding='valid',
                      kernel_initializer='he_normal',
                      activation="relu",
                      name="conv1d_12")(x)
    x = layers.MaxPooling1D(pool_size=3, name="max_pooling1d_12")(x)
    x = layers.Dropout(0.2, name="dropout_22")(x)

    x = Convolution1D(64,
                      7,
                      strides=1,
                      padding='valid',
                      kernel_initializer='he_normal',
                      activation="relu",
                      name="conv1d_13")(x)
    x = layers.MaxPooling1D(pool_size=3, name="max_pooling1d_13")(x)
    x = layers.Dropout(0.2, name="dropout_23")(x)
    
    x = Convolution1D(64,
                      7,
                      strides=1,
                      padding='valid',
                      kernel_initializer='he_normal',
                      activation="relu",
                      name="conv1d_14")(x)
    x = layers.MaxPooling1D(pool_size=3, name="max_pooling1d_14")(x)
    x = layers.Dropout(0.2, name="dropout_24")(x)

    # x = layers.Bidirectional(LSTM(16, return_sequences=True),
    #                          name="bidirectional_6")(x)
    x = layers.Dropout(0.2, name="dropout_25")(x)
    x = layers.Flatten(name="flatten_6")(x)
    x = layers.Dense(128, activation='sigmoid', name="dense_11")(x)
    x = layers.Dropout(0.2, name="dropout_26")(x)
    x = layers.Dense(2, activation='softmax', name="dense_12")(x)

    model = models.Model(inputs, x, name='hs3d_model')
    return model
