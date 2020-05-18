import json as json
import os
import logging
import pprint
from datetime import datetime

from pandas import pd
import numpy as np
import h5py
import pickle

from keras.utils import to_categorical
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Dropout

from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.models import load_model
from keras.models import model_from_json
from keras.models import model_from_yaml

# https://www.youtube.com/watch?v=DPBspKl2epk

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels

from NLPEngine import nlpEngine

import Predict.NLPEngine as en

class NLPData():
    def __init__(self, df, xName, yName):
        self.xName = xName
        self.yName = yName

        self.X = df[self.xName]
        self.y = df[self.yName]

    def traintestsplit(self, prop):
       return train_test_split(self.X, self.y, test_size=prop)

        
    def createTokenizeMatrix(self, X_train, X_test, max_words):
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(X_train)

        # pp = pprint.PrettyPrinter(indent=4,width=120)
        # pp.pprint(tokenizer.to_json())
    
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

        model.s
        return model

    def train_model(self, max_words, max_epochs):

        X_train, X_test, y_train, y_test = self.traintestsplit(0.3) 
        labels = unique_labels(y_test)

        tokenizer, X_train_tokens, X_test_tokens = self.createTokenizeMatrix(X_train, X_test, max_words)
        y_train_cat, y_test_cat, num_classes = self.convertLabelToCategorical(y_train, y_test)

        model = self.create_model(max_words, num_classes)
        model.fit(X_train_tokens, y_train_cat, validation_split=0.1, epochs=max_epochs, batch_size=64, verbose=2)


        score = model.evaluate(X_test_tokens, y_test_cat, batch_size=64, verbose=1)

        pred_class = model.predict_classes(X_test_tokens)
        # classReport = classification_report(y_test_cat, pred_class, target_names=labels)
        tabReport = pd.crosstab(labels[pred_class], y_test,
                      colnames=['Actual'],
                      rownames=["Predicted"],
                      margins=True).to_string()

        ## pp.print(classReport)
        print('*')
        print("****************************************")
        print(tabReport)
        print("****************************************")
        print('*')

        nlp = en.NLPEngine()


        nlp.modelSpec(labels, tokenizer, model)
        nlp.reports({
            'score': { 'Test score':  score[0], 'Test accuracy': score[1]},
            'crossTabReport': tabReport,
        })
        nlp.setHyper({
            'max_words': max_words,
            'max_epochs': max_epochs,     
        })


        return nlp
