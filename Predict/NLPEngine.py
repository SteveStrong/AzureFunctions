import json as json
import os
import logging
import pprint
from datetime import datetime

from pandas import pd
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

# remember to load requirements
# pip install -r requirements.txt

class nlpEngine():
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
