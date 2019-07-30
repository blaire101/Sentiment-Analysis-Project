#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pandas as pd
import ast

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from config import train_data_path
from config import validate_data_path
from config import test_data_path

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd


def load_data_from_json(file_name):
    file_data = open(file_name)

    list_data = []
    while 1:
        line = file_data.readline()
        if not line:
            break
        sample_dict = ast.literal_eval(str(line).replace('\n', ''))
        list_data.append(sample_dict)
    df = pd.DataFrame(list_data)
    return df


def get_accuracy_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def get_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=[1, 0], average='macro')


def get_precision_score(y_true, y_pred):
    return precision_score(y_true, y_pred, labels=[1, 0], average='macro')


def get_recall_score(y_true, y_pred):
    return recall_score(y_true, y_pred, labels=[1, 0], average='macro')


############## data process #############

def get_x_and_y_for_train_dev_test():
    df_train = load_data_from_json(train_data_path)
    x_train, y_train = df_train['sentence'], df_train['label']

    df_dev = load_data_from_json(validate_data_path)
    x_dev, y_dev = df_dev['sentence'], df_dev['label']

    df_test = load_data_from_json(test_data_path)
    x_test, y_test = df_test['sentence'], df_test['label']

    return x_train, y_train, x_dev, y_dev, x_test, y_test


def show_metrics_result(y_dev, y_pred_dev, type_str):
    print("\n=====" + type_str + " run result=====:\n")

    print(type_str + " f1_score: " + str(get_f1_score(y_true=y_dev, y_pred=y_pred_dev)))
    print(type_str + " precision_score: " + str(get_precision_score(y_true=y_dev, y_pred=y_pred_dev)))
    print(type_str + " recall_score: " + str(get_recall_score(y_true=y_dev, y_pred=y_pred_dev)))


##### deep learning data process ########

def get_tokenizer(num_words, texts):
    tokenizer = Tokenizer(num_words=num_words)

    tokenizer.fit_on_texts(texts=texts)

    return tokenizer


def get_train_dev_test_sequences(tokenizer, x_train, x_dev, x_test):
    train_sequences = tokenizer.texts_to_sequences(x_train)
    dev_sequences = tokenizer.texts_to_sequences(x_dev)
    test_sequences = tokenizer.texts_to_sequences(x_test)

    return train_sequences, dev_sequences, test_sequences


def get_padded_sequences(maxlen, data_sequences):
    padded_sequences = pad_sequences(data_sequences, maxlen=maxlen)

    return padded_sequences


def get_train_dev_test_padded_sequences(maxlen, train_sequences, dev_sequences, test_sequences):
    padded_train_sequences = get_padded_sequences(maxlen, train_sequences)

    padded_dev_sequences = get_padded_sequences(maxlen, dev_sequences)

    padded_test_sequences = get_padded_sequences(maxlen, test_sequences)

    return padded_train_sequences, padded_dev_sequences, padded_test_sequences


def get_coefs(word, *arr):
    try:
        return word, np.asarray(arr, dtype='float32')
    except:
        return None, None


def get_embedding_matrix(embedding_data_path, embed_size, tokenizer, num_words):

    # glove.840B.300d.txt
    embeddings_index = dict(get_coefs(*o.strip().split()) for o in
                            open(embedding_data_path))

    values = list(embeddings_index.values())

    all_embs = np.stack(values)

    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    word_index = tokenizer.word_index

    # num_words = MAX_NUM_WORDS
    embedding_matrix = np.random.normal(emb_mean, emb_std, (num_words, embed_size))

    oov = 0

    for word, i in word_index.items():

        if i >= num_words:
            continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            oov += 1

    print(oov)

    return embedding_matrix