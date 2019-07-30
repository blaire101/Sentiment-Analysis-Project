# Sentiment Analysis Project

## Test Description

This task is a traditional NLP task: Text Classification for Sentiment Analysis.

- process raw data to extract features.
- build one or more model (e.g. Logistic Regression, Decision Tree, Neural Network) to train on such data set
- test trained model on some evaluation metric (e.g. Precision, Recall, AUC)

## 2. Development Environment

Depend mainly on list the tools and libraries . For details, please check requirements.txt

## 3. Project Structure

This is a detailed description of the organizational structure of the project.

### 3.1 organization structure

directory structure |  description
------ | ------
data | data files and glove.840B.300d.txt
ipynb | notebook code
model | model files by deep learning
model_best | best model files by a method of deep learning
predict_result | **prediction result for test.json**
images | some pictures for data analysis
**src** | **main code**

### 3.2 code structure

src, core code |  description
------ | ------
-- src/config.py | project configuration information module, mainly including file reading or storage path information
-- src/constant.py | constant variables or infrequently changing variables
-- src/util.py | data processing module, mainly including data reading and processing functions
-- src/model.py | deep learning model definition

### 3.2.1 deep learning model

**src, main train and predict code** | description
------ | ------
-- src/main\_train\_dl_... | model training module, model training process includes data processing, feature extraction, model training, model validation and other steps.
-- src/main\_train\_dl\_1\_rnn\_simple\_and\_predict.py | rnn simple non glove\_embedding
-- src/main\_train\_dl\_2\_rnn\_glove\_embedding\_and\_predict.py | rnn\_glove\_embedding
-- src/main\_train\_dl\_3\_cnn\_and\_predict.py | cnn\_glove\_embedding
-- src/main\_train\_dl\_4\_rcnn\_glove\_embedding\_and\_predict.py | rcnn\_glove\_embedding
-- src/main\_train\_dl\_5\_elmo_like\_and\_predict.py | elmo_like\_glove\_embedding

### 3.2.2 machine learning model

**src, main train and predict code** | description
------ | ------
-- src/main\_train\_ml_... | Model training module, model training process includes data processing, feature extraction, model training, model validation and other steps.
-- src/main\_train\_ml\_1\_lr\_word\_and\_predict.py | logistic regression + word ngram
-- src/main\_train\_ml\_2\_lr\_char\_and\_predict.py | logistic regression + char ngram
-- src/main\_train\_ml\_3\_lr\_word\_char\_and\_predict.py | logistic regression + word\_char ngram

## 4. Instructions

- Prepare pyenv & pip install -r requirement.txt
- config data file storage path in config.py 
- run script: run.sh , The training model is saved and the (precision\_score, recall\_score, f1-score) of the test set will be shown by log.

Tips:

- sh run.sh , only run a best model 

```python
# python src/main_train_dl_4_rcnn_glove_embedding_and_predict.py -ep 10
```

rcnn model metrics result:

```bash
=====test run result=====:

test f1_score: 0.8123589611797799
test precision_score: 0.8125411611403617
test recall_score: 0.8123827392120075
```

other machine learning model, train and predict

```python
# python src/main_train_ml_1_lr_word_and_predict.py
# python src/main_train_ml_2_lr_char_and_predict.py
# python src/main_train_ml_3_lr_word_char_and_predict.py
```

other deep learning model, train and predict

1. when you train the following model, maybe you need to make sure that you have **comment out the prediction code part and metrics code part**.
2. When you have completed the model training, you need to open the comments (prediction code part and metrics code part) and **load the best model** you have created. 
3. when you want to **prediction and evaluation model**, you need to make sure that you have comment out the **train code part** or set parameter
 `-ep 0`

```python
# python src/main_train_dl_1_rnn_simple_and_predict.py -ep 10
# python src/main_train_dl_2_rnn_glove_embedding_and_predict.py -ep 10
# python src/main_train_dl_3_cnn_and_predict.py  -ep 10
# python src/main_train_dl_5_elmo_like_and_predict.py -ep 10
```

## 5. Submit Requirement

The output should include

- prediction result for test.json
- evaluation result for training and predicting performance

## Reference

- [Numpy中stack()，hstack()，vstack()函数详解][1]
- [Understanding LSTM Networks][l2]
- [Understanding Convolutional Neural Networks for NLP][l1]
- [Sentiment analysis on Twitter using word2vec and keras][l3]

[1]: https://blog.csdn.net/csdn15698845876/article/details/73380803
[l1]: http://wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
[l2]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
[l3]: https://ahmedbesbes.com/sentiment-analysis-on-twitter-using-word2vec-and-keras.html
