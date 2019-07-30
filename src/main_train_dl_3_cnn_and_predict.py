#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

from util import get_x_and_y_for_train_dev_test, show_metrics_result

import config
import constant

import util
import model

import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-bs', '--batch_size', type=int, nargs='?',
                        default=256)
    parser.add_argument('-ep', '--epochs', type=int, nargs='?',
                        default=7)

    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs

    # 1. data process
    logger.info("start train...")

    x_train, y_train, x_dev, y_dev, x_test, y_test = get_x_and_y_for_train_dev_test()

    tokenizer = util.get_tokenizer(num_words=constant.MAX_NUM_WORDS, texts=list(x_train)+list(x_dev))

    train_sequences, dev_sequences, test_sequences = util.get_train_dev_test_sequences(tokenizer, x_train, x_dev, x_test)

    padded_train_sequences, padded_dev_sequences, padded_test_sequences = util.get_train_dev_test_padded_sequences(
        maxlen=constant.MAX_LEN,
        train_sequences=train_sequences,
        dev_sequences=dev_sequences,
        test_sequences=test_sequences)

    # 2. embedding_matrix

    from config import glove_embedding_data_path

    num_words = constant.MAX_NUM_WORDS

    embedding_matrix = util.get_embedding_matrix(embedding_data_path=glove_embedding_data_path,
                         embed_size=constant.EMBED_SIZE,
                         tokenizer=tokenizer,
                         num_words=constant.MAX_NUM_WORDS)


    # 3. model
    cnn_model = model.get_cnn_model(max_num_words=constant.MAX_NUM_WORDS,
                                            max_len=constant.MAX_LEN,
                                            embedding_dim_size=constant.EMBED_SIZE,
                                            embedding_matrix=embedding_matrix)

    # 4. train
    filepath = config.model_path + "cnn-ep-{epoch:02d}-val_acc-{val_acc:.4f}.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')


    history = cnn_model.fit(x=padded_train_sequences,
                                y=y_train,
                                validation_data=(padded_dev_sequences, y_dev),
                                batch_size=batch_size,
                                callbacks=[checkpoint],
                                epochs=epochs,
                                verbose=1)

    logger.info("complete model trained!")

    # # 5. predict code
    #
    # model_name = config.best_model_path + 'cnn-weights-improvement-09-0.7740.hdf5'
    #
    # best_cnn_model = load_model(model_name)
    #
    # logger.info("start predict test data")
    # y_pred_cnn = best_cnn_model.predict(padded_test_sequences, verbose=1, batch_size=128)
    #
    # y_pred_cnn = pd.DataFrame(y_pred_cnn, columns=['prediction'])
    #
    # y_pred_cnn['prediction'] = y_pred_cnn['prediction'].map(lambda p: 1 if p >= 0.5 else 0)
    #
    # y_pred_cnn.to_csv(config.predict_result_path + '/y_pred_cnn.csv', index=False)
    #
    # y_pred_cnn = pd.read_csv(config.predict_result_path + '/y_pred_cnn.csv')
    #
    #
    # logger.info("complete predict test data")
    #
    # # 6. metrics code
    #
    # show_metrics_result(y_test, y_pred_cnn, "test")

# =====test run result=====:
#
# test f1_score: 0.776480538473737
# test precision_score: 0.7780036988185705
# test recall_score: 0.7767354596622889