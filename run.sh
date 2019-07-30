#!/bin/bash

########### best model ############

python src/main_train_dl_4_rcnn_glove_embedding_and_predict.py -ep 10

# =====test run result=====:
#
# test f1_score: 0.8123589611797799
# test precision_score: 0.8125411611403617
# test recall_score: 0.8123827392120075

######### other model, train and predict #############

# python src/main_train_ml_1_lr_word_and_predict.py
# python src/main_train_ml_2_lr_char_and_predict.py
# python src/main_train_ml_3_lr_word_char_and_predict.py

# ...train....

# python src/main_train_dl_1_rnn_simple_and_predict.py -ep 10
# python src/main_train_dl_2_rnn_glove_embedding_and_predict.py -ep 10
# python src/main_train_dl_3_cnn_and_predict.py  -ep 10
# python src/main_train_dl_4_rcnn_glove_embedding_and_predict.py -ep 10
# python src/main_train_dl_5_elmo_like_and_predict.py -ep 10

# ...prediction and evaluation....
#
# python src/main_train_dl_1_rnn_simple_and_predict.py -ep 0
# python src/main_train_dl_2_rnn_glove_embedding_and_predict.py -ep 0
# python src/main_train_dl_3_cnn_and_predict.py  -ep 0
# python src/main_train_dl_4_rcnn_glove_embedding_and_predict.py -ep 0
# python src/main_train_dl_5_elmo_like_and_predict.py -ep 0