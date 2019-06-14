import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import csv

#data_path = '../data/preprocessed/13_01_000002/1.txt'

def find_entity(data_path):

    with open(data_path) as f:
            #data = f.readlines()
        data = " ".join(line.strip() for line in f)
            #text_data.append(data)

    #nlp = spacy.load("en_core_web_sm")
    nlp = en_core_web_sm.load()
    doc = nlp(data)
    entity = [(X.text, X.label_) for X in doc.ents]
    return entity

def export_entity_csv(data_path, csv_dir):
    data = find_entity(data_path)


    with open(csv_dir + '/entity.csv','w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['words','entity'])
        for row in data:
            csv_out.writerow(row)
#entity = find_entity(data_path)
#print(entity)    
