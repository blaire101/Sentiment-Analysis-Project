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

    x_train, y_train, x_dev, y_dev, x_test, y_test = get_x_and_y_for_train_dev_test()

    # 2. feature,  TFIDF, ngram
    vectorizer_char = TfidfVectorizer(max_features=40000,
                                      min_df=5,
                                      max_df=0.5,
                                      analyzer='char',
                                      ngram_range=(1, 4))

    vectorizer_char.fit(x_train)

    tfidf_matrix_char_train = vectorizer_char.transform(x_train)
    tfidf_matrix_char_dev = vectorizer_char.transform(x_dev)
    tfidf_matrix_char_test = vectorizer_char.transform(x_test)

    # 3. model
    lr_char = LogisticRegression(solver='sag', verbose=2)

    # 4. train
    lr_char.fit(tfidf_matrix_char_train, y_train)

    # 5. predict
    y_pred_dev = lr_char.predict(tfidf_matrix_char_dev)

    y_pred_test = lr_char.predict(tfidf_matrix_char_test)

    # 6. metrics
    show_metrics_result(y_dev, y_pred_dev, "dev")

    show_metrics_result(y_test, y_pred_test, "test")

# =====dev run result=====:
#
# dev f1_score: 0.755131546242257
# dev precision_score: 0.7555288534663706
# dev recall_score: 0.7552083333333334
#
# =====test run result=====:
#
# test f1_score: 0.7570097735063609
# test precision_score: 0.7571451712869055
# test recall_score: 0.7570356472795496