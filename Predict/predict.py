import logging
import json
import sys
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

class PayloadWrapper:
    def wrapList(self, payload, message=''):
        result = {
            'hasErrors': len(message) > 0,
            'message': message,
            'payloadCount': len(payload),
            'payload': payload,
        }
        return json.dumps(result, indent=4, default=str)

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


    def predict(self, text:str):
     
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

        return result

    def load(self, name:str):
        # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
        pp = pprint.PrettyPrinter(indent=4,width=120)

        directory = f"Predict\\NLP_{name}\\"
        fileName = f"{name}.json"

        with open(directory + fileName) as infile:
            saveSpec = json.load(infile)
            

        with open(directory + saveSpec['tokens'], 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        with open(directory + saveSpec['labels'], 'rb') as handle:
            self.labels = pickle.load(handle)

        self.model = load_model(directory + saveSpec['model'])
        self.model.load_weights(directory + saveSpec['weights'])

        # https: // stackoverflow.com / questions / 43971649 / dump - weights - of - cnn - in - json - using - keras


        # # serialize model to JSON
        # model_json = model.to_json()
        # with open("model.json", "w") as json_file:
        #     json_file.write(model_json)
        # # serialize weights to HDF5
        # model.save_weights("model.h5")
        # print("Saved model to disk")
        
        # # later...
        
        # # load json and create model
        # json_file = open('model.json', 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # loaded_model = model_from_json(loaded_model_json)
        # # load weights into new model
        # loaded_model.load_weights("model.h5")
        # print("Loaded model from disk")

        return saveSpec



def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        logging.info('STEVE!!!! function predicting request.')

        nlp = NLPHyperEngine()
        logging.info(nlp)

        time = datetime.utcnow

        text = f"The {time} Veteran did not have a psychiatric disorder in service that was unrelated to the use of drugs."
        
        nlp.load("version1")
        result = nlp.predict(text)
        pw = PayloadWrapper()
    
        return func.HttpResponse(
            body= pw.wrapList([result]),
            status_code=200
        )

    except BaseException as e:
        pw = PayloadWrapper()
        message = "sorry"
        return func.HttpResponse(
            body= pw.wrapList([],message),
             status_code=400
        )
