import os
import json
import logging
import pandas as pd
import numpy as np

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
import pprint

from keras.utils import to_categorical
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Dropout

from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.models import load_model
from keras.models import model_from_json
from keras.models import model_from_yaml


# monkey patch courtesy of
# https://github.com/noirbizarre/flask-restplus/issues/54
# so that /swagger.json is served over https


# remember to load requirements
# pip install -r requirements.txt

from NLPData import NLPData
from NLPEngine import NLPEngine

def modelTrainer(dataFile, xName, yName):

    df = pd.read_pickle(dataFile)

    data = NLPData(df, xName, yName)

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
