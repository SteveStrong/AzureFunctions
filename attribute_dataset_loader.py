# This pickle creator function was developed for the Legal Apprentice workflow,
# written by John Milne, 10/17/2019

# This function takes the JSON-formatted data in /Data that the Legal
# Apprentice data starts as, throws it into a dataframe and then pickles that
# to the /Pickle directory.

# The assumption here is that the data stored at "~/.Data/" hasn't previously
# been pickled and the running of this function is to create the pickle file
# of the JSON-formatted NEW data, which is then moved to the /Pickle directory;
# thus, the "./Data/" directory only contains that data which needs to be
# pickled and nothing else and any data in the /Pickle directory has either
# been moved or renamed so as not to overwrite any previous work (that should
# not be overwritten).

def legal_apprentice_pickler():
    
    # Imports of import.
    import json
    import os
    import pandas as pd  
        

    # Getting the list of files in <data_path>:
    data_path = './data/CURATED-BVA-Decisions/'
    list_of_files = os.listdir(data_path)

    # ...and creating new lists for the texts of the sentences...
    df_rows = []
    df_attributions = []
    index = 0
    
    print(len(list_of_files))
    # Using a for-loop to iterate over the filenames...
    for filename in list_of_files:
        print ( filename )

        # ... and opening the given filename...
        file = open(data_path + filename)
        
        # ...using the json file loader to translate the json data...
        data = json.load(file)

        # ...and adding the sentences to those new lists...
        for sentence in data['sentences']:
            sentenceNumber = int(sentence['sentenceNumber'])
            paragraphNumber = int(sentence['paragraphNumber'])
            
            # ...assigning the attributions values to a python list
            attribList = sentence.get('attributions')
            # ...converting the python list to a panda dataframe
            #attDF = pd.DataFrame(attribList)
            # ...printing every attribution dataframe in the terminal as generated, but see line 79 for printing only the last dataframe
            #print(attDataFrame)
            
            if attribList is not None:
                for attr in attribList:
                    cue = attr.get('cue')
                    cue = "" if cue is None else cue
                    type = attr.get('type')
                    type = "" if type is None else type
                    subj = attr.get('subject')
                    subj = "" if subj is None else subj
                    obj = attr.get('object')
                    obj = "" if obj is None else obj
                    pola = attr.get('polarity')
                    pola = "" if pola is None else pola
                    attrecord = {
                        'attriType': type,
                        'sentRhetClass': sentence['rhetClass'],
                        'sentID' : sentence['sentID'],
                        'attriSubject': subj,
                        'attriCue': cue,
                        'attriObject': obj,
                        'attriPolarity': pola,
                        'sentText': sentence['text'],
                    }
                    df_attributions.append(attrecord)

            
            record = {
                'index': index,
                'sentID' : sentence['sentID'],
                'caseNumber' : sentence['caseNumber'],
                'sentenceNumber' : sentenceNumber,
                'paragraphNumber' : paragraphNumber,
                'isFirst': sentenceNumber == 1,
                'isLast': False,
                'rhetClass': sentence['rhetClass'],
                'text': sentence['text'],
            }
            index += 1
            df_rows.append(record)
            

            
    # Creating the dataframe into which all of the files will be stored:
    # ...and adding those to the previously instantiated dataframe:
    df = pd.DataFrame(df_rows)


                
    # Pickling the dataframe:
    df.to_pickle("BVAsentence.pkl")
    df.to_csv("BVAsentence.csv",sep='|')

    # Pickling the attributions:
    dfAtt = pd.DataFrame(df_attributions)
    dfAtt.to_pickle("BVAattribute.pkl")
    dfAtt.to_csv("BVAattribute.csv",sep='|')
    
    # Now to pass the fact that this has completed as the return statement:
    pickled = 'Done'
    
    return pickled

legal_apprentice_pickler()