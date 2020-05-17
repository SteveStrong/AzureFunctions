import numpy as np
import os
import json
import logging
import pandas as pd

# https://www.youtube.com/watch?v=DPBspKl2epk

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels

# import tensorflow as tf
from tensorflow import keras
import numpy as np
import json as json
import pprint

from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Dropout

from keras.preprocessing.text import Tokenizer, tokenizer_from_json


from flask import Flask
app = Flask(__name__)

# monkey patch courtesy of
# https://github.com/noirbizarre/flask-restplus/issues/54
# so that /swagger.json is served over https


app.gLabels = None 
app.gTokenizer = None
app.gModel = None

# remember to load requirements
# pip install -r requirements.txt


class Data():
    def __init__(self, df):
        self.X = df['attriCue']
        self.y = df['sentRhetClass']

    def traintestsplit(self, prop):
       return train_test_split(self.X, self.y, test_size=prop)

        
    def createTokenizeMatrix(self, X_train, X_test, max_words):
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(X_train)
    
        #Convert train and test sets to tokens
        X_train_tokens = tokenizer.texts_to_matrix(X_train, mode='tfidf')
        X_test_tokens = tokenizer.texts_to_matrix(X_test, mode='tfidf')

        return tokenizer, X_train_tokens, X_test_tokens


    def convertLabelToCategorical(self, y_train, y_test):
        #Convert labels to a one-hot representation
        encoder = LabelEncoder()
        encoder.fit(y_train)
        y_train_encode = encoder.transform(y_train)
        y_test_encode = encoder.transform(y_test)

        num_classes = np.max(y_train_encode)+1
        y_train_cat = to_categorical(y_train_encode, num_classes)
        y_test_cat = to_categorical(y_test_encode, num_classes)   

        return y_train_cat, y_test_cat, num_classes


    def create_model(self, max_words, num_classes):
        model = Sequential()

        model.add(Dense(512,input_shape=(max_words,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        return model

def modelTrainer():

    df = pd.read_pickle("BVAattribute.pkl")

    data = Data(df)

    X_train, X_test, y_train, y_test = data.traintestsplit(0.3) 

    labels = unique_labels(y_test)
    
    max_words = 3000
    tokenizer, X_train_tokens, X_test_tokens = data.createTokenizeMatrix(X_train, X_test, max_words)
    y_train_cat, y_test_cat, num_classes = data.convertLabelToCategorical(y_train, y_test)

    model = data.create_model(max_words, num_classes);
    
    model.fit(X_train_tokens, y_train_cat,
              validation_split=0.1, epochs=22, batch_size=64, verbose=2)

    app.gLabels = labels
    app.gTokenizer = tokenizer
    app.gModel = model

    return True


def classifySentence(text:str):
    # new_sentence = ["Veteran had a disorder in service"]
    # new_sentence = ["The most probative evidence fails to link the Veteran's claimed acquired psychiatric disorder, including PTSD, to active service or to his service-connected residuals of frostbite."]
    
    new_sentence = [text]
    label = ''

    seq = app.gTokenizer.texts_to_matrix(new_sentence)
    pred_sent = app.gModel.predict(seq)
    pred_class_sent = app.gModel.predict_classes(seq)
    label = app.gLabels[pred_class_sent][0]

    result = {
                'text': text,
                'classification': label
             }
 

    return result


def predictSentence(text:str):
     
    new_sentence = [text]
    label = ''

    seq = app.gTokenizer.texts_to_matrix(new_sentence)
    pred_sent = app.gModel.predict(seq)
    pred_class_sent = app.gModel.predict_classes(seq)
    label = app.gLabels[pred_class_sent][0]

    items = {app.gLabels[i]: str(pred_sent[0][i]) for i in range(len(app.gLabels))} 

    result = {
                'text': text,
                'classification': label,
                'predictions': items
             }

    return result



def startup():
    logging.warning('getting started')
    if ( app.gModel == None):
        modelTrainer()

    text = "The Board finds that the Veteran had an in-service stressor."

    print("-------------------------------")
    print(text)
    print("-------------------------------")

    result = classifySentence(text)
    pprint.pprint(result)
    print("-------------------------------")

    result = predictSentence(text)
    pprint.pprint(result)

startup()
