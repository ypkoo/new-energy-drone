#!/usr/bin/env python
#coding=utf8
from keras.models import Sequential, Model
from keras.layers import Reshape, Dropout, Conv1D, Convolution1D, Dense, MaxPooling1D, Flatten, Activation, Concatenate, Input, concatenate, LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import *
import keras.backend as K
import math
from keras.utils import multi_gpu_model



# Define metric
def mean_acc(y_true, y_pred):
    return K.mean(1-K.abs(y_true-y_pred)/y_true)


# Models
def base_model(input_dim, output_dim=1):
    model = Sequential()

    model.add(Dense(12, input_dim=input_dim, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(15, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(15, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(output_dim, kernel_initializer='uniform', activation='relu'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    return model



def flexible_model_koo(weights, input_dim, output_dim=1):

    model = Sequential()

    model.add(Dense(100, input_dim=input_dim, kernel_initializer='uniform'))
    # model.add(Dropout(FLAGS.dropout_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    
    for weight in weights:
        model.add(Dense(weight, kernel_initializer='uniform'))
        # model.add(Dropout(FLAGS.dropout_rate))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        

    model.add(Dense(output_dim, kernel_initializer='uniform'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', mean_acc])
    model.summary()

    return model




def lstm(input_shape):

    lstm_input = Input(shape=input_shape)

    lstm = LSTM(32)(lstm_input)

    x = Dense(32, activation='relu')(lstm)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    # x = Dense(64, activation='relu')(x)

    output = Dense(1)(x)

    model = Model(lstm_input, output)

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', mean_acc])
    model.summary()

    return model

def gru(input_shape):

    gru_input = Input(shape=input_shape)

    gru = GRU(32)(gru_input)

    x = Dense(64, activation='relu')(gru)
    x = Dense(64, activation='relu')(x)
    # x = Dense(64, activation='relu')(x)

    output = Dense(1)(x)

    model = Model(gru_input, output)

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', mean_acc])
    model.summary()

    return model