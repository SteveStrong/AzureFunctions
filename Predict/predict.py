import logging
import json as json
import os
import logging
import pprint
from datetime import datetime

# from pandas import pd
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

import azure.functions as func

class NLPHyperEngine():
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

    def load(self, name:str):
        # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
        pp = pprint.PrettyPrinter(indent=4,width=120)

        directory = os.getcwd() +'\\NLP_' + name + '\\'
        fileName = directory + name + '.json'

        pp.pprint(fileName)
        logging(fileName)


        with open(fileName) as infile:
            saveSpec = json.load(infile)
            
        logging(saveSpec)
        pp.pprint(saveSpec)

        self.model = load_model(directory + saveSpec['model'])
        self.model.load_weights(directory + saveSpec['weights'])

        with open(directory + saveSpec['tokens'], 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        with open(directory + saveSpec['labels'], 'rb') as handle:
            self.labels = pickle.load(handle)

        return saveSpec



def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        logging.info('STEVE!!!! function predicting request.')

        nlp = NLPHyperEngine()
        logging.info(nlp)

        time = datetime.utcnow

        text = f"The {time} Veteran did not have a psychiatric disorder in service that was unrelated to the use of drugs."
        
        #  nlp.load("version1")
        result = text
        #  result = nlp.predict(text, print=True)
    
        return func.HttpResponse(
            body=result,
            status_code=200
        )

    except:
        return func.HttpResponse(
             "Please pass a name param  on the query string or in the request body",
             status_code=400
        )
