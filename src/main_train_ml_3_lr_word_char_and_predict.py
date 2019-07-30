#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from util import get_x_and_y_for_train_dev_test, show_metrics_result

import scipy
from scipy.sparse import hstack

def get_tfidf_matrix_word_train_and_dev(x_train, x_dev, x_test):
    vectorizer_word = TfidfVectorizer(max_features=40000,
                                      min_df=5,
                                      max_df=0.5,
                                      analyzer='word',
                                      stop_words='english',
                                      ngram_range=(1, 2))
    vectorizer_word.fit(x_train)
    # 将这个矩阵作为输入，用transformer.fit_transform(词频矩阵)得到TF-IDF权重矩阵
    tfidf_matrix_word_train = vectorizer_word.transform(x_train)
    tfidf_matrix_word_dev = vectorizer_word.transform(x_dev)
    tfidf_matrix_word_test = vectorizer_word.transform(x_test)

    return tfidf_matrix_word_train, tfidf_matrix_word_dev, tfidf_matrix_word_test


def get_tfidf_matrix_char_train_and_dev(x_train, x_dev, x_test):

    vectorizer_char = TfidfVectorizer(max_features=40000,
                                      min_df=5,
                                      max_df=0.5,
                                      analyzer='char',
                                      ngram_range=(1, 4))
    vectorizer_char.fit(x_train)
    # vectorizer_char.fit(x_train)
    tfidf_matrix_char_train = vectorizer_char.transform(x_train)
    tfidf_matrix_char_dev = vectorizer_char.transform(x_dev)
    tfidf_matrix_char_test = vectorizer_char.transform(x_test)

    return tfidf_matrix_char_train, tfidf_matrix_char_dev, tfidf_matrix_char_test

if __name__ == '__main__':

    # 1. data x, y

    x_train, y_train, x_dev, y_dev, x_test, y_test = get_x_and_y_for_train_dev_test()


    # 2. feature, TFIDF, ngram

    tfidf_matrix_word_train, tfidf_matrix_word_dev, tfidf_matrix_word_test = get_tfidf_matrix_word_train_and_dev(x_train, x_dev, x_test)

    tfidf_matrix_char_train, tfidf_matrix_char_dev, tfidf_matrix_char_test = get_tfidf_matrix_char_train_and_dev(x_train, x_dev, x_test)

    tfidf_matrix_word_char_train = hstack((tfidf_matrix_word_train, tfidf_matrix_char_train))

    tfidf_matrix_word_char_dev = hstack((tfidf_matrix_word_dev, tfidf_matrix_char_dev))

    tfidf_matrix_word_char_test = hstack((tfidf_matrix_word_test, tfidf_matrix_char_test))

    # 3. model

    lr_word_char = LogisticRegression(solver='sag', verbose=2)

    # 4. train

    lr_word_char.fit(tfidf_matrix_word_char_train, y_train)

    # 5. predict
    y_pred_dev = lr_word_char.predict(tfidf_matrix_word_char_dev)

    y_pred_test = lr_word_char.predict(tfidf_matrix_word_char_test)

    # 6. metrics
    show_metrics_result(y_dev, y_pred_dev, "dev")

    show_metrics_result(y_test, y_pred_test, "test")

# =====dev run result=====:
#
# dev f1_score: 0.7624907222938396
# dev precision_score: 0.7625410220346929
# dev recall_score: 0.7625
#
# =====test run result=====:
#
# test f1_score: 0.7747950761950367
# test precision_score: 0.7751731186016598
# test recall_score: 0.774859287054409