#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from keras.models import Model

from keras.optimizers import Adam

from keras.layers import Input, Dense, Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.layers import SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.layers import Activation



def get_simple_rnn_model(max_num_words, max_len, embedding_dim_size):

    embedding_matrix = np.random.random((max_num_words, embedding_dim_size))

    inp = Input(shape=(max_len,))

    x = Embedding(input_dim=max_num_words, output_dim=embedding_dim_size, input_length=max_len,
                  weights=[embedding_matrix], trainable=True)(inp)

    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(GRU(100, return_sequences=True))(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    conc = concatenate([avg_pool, max_pool])

    outp = Dense(1, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def get_rnn_model_with_glove_embeddings(max_num_words, max_len, embedding_dim_size, embedding_matrix):

    embedding_dim = embedding_dim_size

    inp = Input(shape=(max_len,))

    x = Embedding(max_num_words, embedding_dim, weights=[embedding_matrix],
                  input_length=max_len, trainable=True)(inp)

    x = SpatialDropout1D(0.3)(x)

    x = Bidirectional(GRU(100, return_sequences=True))(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    conc = concatenate([avg_pool, max_pool])
    outp = Dense(1, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def get_rnn_cnn_model(max_num_words, max_len, embedding_dim_size, embedding_matrix):

    inp = Input(shape=(max_len,))
    x = Embedding(max_num_words, embedding_dim_size, weights=[embedding_matrix],
                  input_length=max_len, trainable=True)(inp)

    x = SpatialDropout1D(0.3)(x)

    x = Bidirectional(GRU(100, return_sequences=True))(x)

    x = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    conc = concatenate([avg_pool, max_pool])

    outp = Dense(1, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def get_cnn_model(max_num_words, max_len, embedding_dim_size, embedding_matrix):
    embedding_dim = embedding_dim_size

    filter_sizes = [2, 3, 5]
    num_filters = 256
    drop = 0.3

    inputs = Input(shape=(max_len,), dtype='int32')

    embedding = Embedding(input_dim=max_num_words,
                          output_dim=embedding_dim,
                          weights=[embedding_matrix],
                          input_length=max_len,
                          trainable=True)(inputs)

    reshape = Reshape((max_len, embedding_dim, 1))(embedding)

    conv_0 = Conv2D(num_filters,
                    kernel_size=(filter_sizes[0], embedding_dim),
                    padding='valid', kernel_initializer='normal',
                    activation='relu')(reshape)

    conv_1 = Conv2D(num_filters,
                    kernel_size=(filter_sizes[1], embedding_dim),
                    padding='valid', kernel_initializer='normal',
                    activation='relu')(reshape)
    conv_2 = Conv2D(num_filters,
                    kernel_size=(filter_sizes[2], embedding_dim),
                    padding='valid', kernel_initializer='normal',
                    activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(max_len - filter_sizes[0] + 1, 1),
                          strides=(1, 1), padding='valid')(conv_0)

    maxpool_1 = MaxPool2D(pool_size=(max_len - filter_sizes[1] + 1, 1),
                          strides=(1, 1), padding='valid')(conv_1)

    maxpool_2 = MaxPool2D(pool_size=(max_len - filter_sizes[2] + 1, 1),
                          strides=(1, 1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)(
        [maxpool_0, maxpool_1, maxpool_2])

    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=1, activation='sigmoid')(dropout)

    model = Model(inputs=inputs, outputs=output)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_elmo_like(max_num_words, max_len, embedding_dim_size, embedding_matrix):

    inp = Input(shape=(max_len,))

    emb_layer1 = Embedding(input_dim=max_num_words, output_dim=embedding_dim_size,
                  weights=[embedding_matrix], trainable=False)

    emb_layer2 = Embedding(input_dim=max_num_words, output_dim=embedding_dim_size, trainable=True)

    x1 = emb_layer1(inp)
    x1 = Activation('tanh')(x1)

    x2 = emb_layer2(inp)
    x2 = Activation('tanh')(x2)

    x_emb = concatenate([x1, x2])

    gru_0_output = Bidirectional(GRU(80, return_sequences=True))(x_emb)
    gru_1_output = Bidirectional(GRU(80, return_sequences=True))(gru_0_output)

    x_emb = concatenate([x_emb, gru_0_output, gru_1_output])

    x_emb = Dropout(0.2)(x_emb)

    x_emb = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x_emb)

    avg_pool = GlobalAveragePooling1D()(x_emb)
    max_pool = GlobalMaxPooling1D()(x_emb)

    conc = concatenate([avg_pool, max_pool])

    outp = Dense(1, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
