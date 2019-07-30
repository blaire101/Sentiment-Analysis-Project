#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from util import get_x_and_y_for_train_dev_test, show_metrics_result


if __name__ == '__main__':

    # 1. data x, y
    logger.info("start...")

    x_train, y_train, x_dev, y_dev, x_test, y_test = get_x_and_y_for_train_dev_test()

    # 2. feature,  TFIDF, ngram
    vectorizer_word = TfidfVectorizer(max_features=40000,
                                      min_df=5,
                                      max_df=0.5,
                                      analyzer='word',
                                      stop_words='english',
                                      ngram_range=(1, 2))

    vectorizer_word.fit(x_train)

    tfidf_matrix_word_train = vectorizer_word.transform(x_train)
    tfidf_matrix_word_dev = vectorizer_word.transform(x_dev)
    tfidf_matrix_word_test = vectorizer_word.transform(x_test)

    # 3. model
    lr_word = LogisticRegression(solver='sag', verbose=2)

    # 4. train
    lr_word.fit(tfidf_matrix_word_train, y_train)

    # 5. predict

    y_pred_word_dev = lr_word.predict(tfidf_matrix_word_dev)

    y_pred_word_test = lr_word.predict(tfidf_matrix_word_test)

    # 6. metrics
    show_metrics_result(y_dev, y_pred_word_dev, "dev")

    show_metrics_result(y_test, y_pred_word_test, "test")

# =====dev run result=====:
#
# dev f1_score: 0.7384164948722207
# dev precision_score: 0.7389991259311443
# dev recall_score: 0.7385416666666667

# =====test run result=====:

# test f1_score: 0.7401498651389751
# test precision_score: 0.7401509391456169
# test recall_score: 0.7401500938086304