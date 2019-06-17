import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import csv


def find_entity(data_path):
    with open(data_path) as f:
        data = " ".join(line.strip() for line in f)
    #nlp = spacy.load("en_core_web_sm")
    nlp = en_core_web_sm.load()
    doc = nlp(data)
    entity = [(X.text, X.label_) for X in doc.ents]
    return entity


def export_entity_csv(data_path, csv_dir):
"""Find entity and returns a sublist of [text, label]"""
    data = find_entity(data_path)
    with open(csv_dir + '/entity.csv','w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['words','entity'])
        for row in data:
            csv_out.writerow(row)
