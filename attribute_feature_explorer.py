import numpy as np
import os
import json
import logging
import pandas as pd

import h5py
import pickle
from datetime import datetime

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
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Dropout

from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.models import load_model
from keras.models import model_from_json
from keras.models import model_from_yaml


from flask import Flask
app = Flask(__name__)

# monkey patch courtesy of
# https://github.com/noirbizarre/flask-restplus/issues/54
# so that /swagger.json is served over https


# remember to load requirements
# pip install -r requirements.txt

class NLPEngine():
    def modelSpec(self, labels, tokenizer, model):
        self.labels = labels
        self.tokenizer = tokenizer
        self.model = model

    def setSpec(self, dataFile, xName, yName):
        self.dataFile = dataFile
        self.xName = xName
        self.yName = yName

    def setHyper(self, params):
        self.hyperParams = params

    def reports(self, report):
        self.report = report


    def predict(self, text:str, print:bool=True):
     
        sentence = [text]

        seq = self.tokenizer.texts_to_matrix(sentence)
        pred_sent = self.model.predict(seq)
        pred_class_sent = self.model.predict_classes(seq)
        label = self.labels[pred_class_sent][0]

        items = {self.labels[i]: str(pred_sent[0][i]) for i in range(len(self.labels))} 

        result = {
                    'text': text,
                    'classification': label,
                    'predictions': items
                }

        if print:
            pp = pprint.PrettyPrinter(indent=4,width=120)
            pp.pprint(result)

        return result


    def save(self, name:str):
        # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model

        # datetime object containing current date and time
        now = datetime.now()

        directory = 'NLP_' + name + '/'

        try:
            os.mkdir(directory)
        except OSError:
            print ("Creation of the directory %s failed" % directory)
        else:
            print ("Successfully created the directory %s " % directory)

        saveSpec = {
            'created': now.strftime("%d/%m/%Y %H:%M:%S"),
            'dataFile': self.dataFile,
            'xName': self.xName,
            'yName': self.yName,
            'name': name + '.json',
            'hyperParams': self.hyperParams,
            'report': self.report,
            'dataframe': name + 'DataSet.pkl',
            'dataset': name + 'DataSet.csv',
            'model': name + 'Model.h5',
            'weights': name + 'Weights.h5',
            'tokens': name + 'Tokenizer.pkl',
            'labels': name + 'Label.pkl',
        }

        # https://www.w3schools.com/python/python_json.asp

        with open(directory + saveSpec['name'], 'w') as outfile:
            json.dump(saveSpec, outfile)

        # Pickling the dataframe:
        df = pd.read_pickle(self.dataFile)
        if ( df is not None ):
            df.to_pickle(directory + saveSpec['dataframe']) 
            df.to_csv(directory + saveSpec['dataset'], sep='|') 


        self.model.save(directory + saveSpec['model'])  # creates a HDF5 file 'my_model.h5'
        self.model.save_weights(directory + saveSpec['weights'])

        with open(directory + saveSpec['tokens'], 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(directory + saveSpec['labels'], 'wb') as handle:
            pickle.dump(self.labels, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def load(self, name:str):
        # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
        pp = pprint.PrettyPrinter(indent=4,width=120)

        directory = os.getcwd() +'\\NLP_' + name + '\\'
        fileName = directory + name + '.json'

        pp.pprint(fileName)


        with open(fileName) as infile:
            saveSpec = json.load(infile)
            
        pp.pprint(saveSpec)

        self.model = load_model(directory + saveSpec['model'])
        self.model.load_weights(directory + saveSpec['weights'])

        with open(directory + saveSpec['tokens'], 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        with open(directory + saveSpec['labels'], 'rb') as handle:
            self.labels = pickle.load(handle)

        return saveSpec

class Data():
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

        nlp = NLPEngine()


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

def modelTrainer(dataFile, xName, yName):

    df = pd.read_pickle(dataFile)

    data = Data(df, xName, yName)

    max_words = 3000
    max_epochs = 10


    nlp = data.train_model(max_words,max_epochs)
    nlp.setSpec(dataFile, xName, yName);

    return nlp




def startup():

    nlp1 = modelTrainer("BVAattribute.pkl", 'sentText', 'sentRhetClass')
    nlp1.save("version1")

    nlp2 = modelTrainer("BVAattribute.pkl", 'attriCue', 'attriType')
    nlp2.save("version2")

    nlp3 = NLPEngine();
    nlp3.load("version2")



    print("-------------------------------")

    text = "4. The Veteran did not have a psychiatric disorder in service that was unrelated to the use of drugs."
    
    print("______________")
    nlp2.predict(text,print=True)
    print("______________")
    nlp3.predict(text,print=True)
    print("______________")

    print("-------------------------------")



startup()
