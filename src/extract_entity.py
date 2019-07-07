import spacy
import en_core_web_sm
import csv
import pandas as pd
import tensorflow as tf
from neuroner import neuromodel
import os
import shutil
import glob
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import fuzzyset

def clean_process_name_file(name_file):
    df = pd.read_csv(name_file,header = 0)
    df['name'] = df['name heading'].str.replace('\d+', '')
    df['name'] = df['name'].str.replace(', -', '')
    df['name'] = df['name'].str.split(',')
    for i in range(len(df)):
        #print(df['name'].iloc[i])
        if len(df['name'].iloc[i])==1:
            df['name'].iloc[i] = df['name'].iloc[i][0]
            df['name'].iloc[i] = df['name'].iloc[i].strip()
        else:
            #len(df['name'].iloc[i])==2:
            df['name'].iloc[i] = df['name'].iloc[i][1]+' ' + df['name'].iloc[i][0]
            df['name'].iloc[i] = df['name'].iloc[i].strip()
            
    df['begin'] = df['name heading'].str.extract('(\d\d\d\d)', expand=True)
    df['end'] = df['name heading'].str.extract('(-\d\d\d\d)', expand=True)
    df['end'] = df['end'].str.replace('-','')
    #print('before', len(df))
    df = df[df.name != 'Mass.) Vigilance Committee (Boston']
    #print('after',len(df))
    #df.head()
    return df

def return_likely_year_match_df(match_df):
    if len(match_df) > 1:
        #person most likely should be at least 10 years old in 1861
        ind1 = pd.to_numeric(match_df['begin'],errors='coerce') < 1851
        if len(match_df[ind1]) >= 1:
            match_df = match_df[ind1]            
        #person most likely should be alive in 1861
        #print(len(ind1),len(match_df))
        ind2 = pd.to_numeric(match_df['end'],errors='coerce') > 1861
        if len(match_df[ind2]) >= 1:
            match_df = match_df[ind2]
        #print(len(ind1),len(match_df))
        if len(match_df) > 1:
            match_df = match_df.iloc[0:1]
    return match_df

def run_ner(issues_path, model_path, embedding_file):
    #spacy.load('en')
    text_folder = issues_path
    out_folder = './output'
    ner_folder = out_folder + '/ner'
    if os.path.isdir(ner_folder):
        shutil.rmtree(ner_folder)
    else:
        os.makedirs(ner_folder)
    predict_folder = ner_folder + '/' + 'deploy'
    if not os.path.isdir(predict_folder):
        os.makedirs(predict_folder)
    src_files = os.listdir(text_folder)
    print(src_files)
    for file_name in src_files:
        full_file_name = os.path.join(text_folder, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, predict_folder)
    #model_path = '../model/conll_2003_en'
    #embedding_file = '../model/glove.6B.100d.txt'
    nn = neuromodel.NeuroNER(train_model=False, use_pretrained_model=True, 
                             dataset_text_folder=ner_folder,
                             pretrained_model_folder=model_path,
                             token_pretrained_embedding_filepath=embedding_file, 
                             output_folder=ner_folder + '/annotation')
    nn.fit()
    
def collect_ann_files():
    output_ann_folder = './output/ner/annotation'
    dirs = os.listdir(output_ann_folder)
    #latest directory
    dest = './output/ner/annotation/' + dirs.pop() + '/brat/deploy'
    ann_files = glob.glob(dest + '/*.ann')
    ann_file_names = []
    for ann_file in ann_files:
        tmp = ann_file.split('/')
        #[:-4] for excluding .ann
        ann_file_names.append(tmp.pop()[:-4])
    return ann_files, ann_file_names
    

def get_name_list_per_issue(ann_file):
    with open(ann_file) as f:
        data = f.read().splitlines()

    new_data = []
    for line in data:
        line = line.replace('\t',' ')
        new = line.split(' ')
        new_data.append(new)
        
    results = [' '.join(t[4:]) for t in new_data if t[1] == 'PER']
    results_dedupe = list(process.dedupe(results, threshold=80))
    #print(len(results_dedupe))
    return results_dedupe


def match_name_list(results_dedupe, df):
    name_list_all = []
    name_list_highmatch = []
    fz = fuzzyset.FuzzySet()
    terms=df['name'].tolist()
    #print(len(terms))
    #Create a list of terms we would like to match against in a fuzzy way
    for l in terms:
        fz.add(l)
     
    #Now see if our sample term fuzzy matches any of those specified terms
    for name in results_dedupe:
        sample_term = name
        #matches is a list of tuples (prob, name)
        matches = fz.get(sample_term)
        if matches:
            max_match = max(matches, key=lambda x:x[0])
        else:
            max_match = None
        #match_df = df[df['name'].str.match(matches)]
        #print('-------------------')
        #print(name, matches, max_match)
        #print(name, matches)
        if max_match:
            match_df = df[df['name'].str.match(max_match[1])]
            #print(match_df)
            if len(match_df)>=1:
                match_df = return_likely_year_match_df(match_df)
                #print(len(match_df))
                #print(match_df)
                name_list_all.append([name, match_df.iloc[0]['name heading'],match_df.iloc[0]['URI'],max_match[0]])
                #select = [each for each in matches if each[0]>0.8]
                #print(select)
                if max_match[0] > 0.85:
                #if select :
                    name_list_highmatch.append([name, match_df.iloc[0]['name heading'],match_df.iloc[0]['URI'],max_match[0]])
            else:
                name_list_all.append([name,'','',''])
                #name_list_highmatch.append([name,'','',''])
        else:
            name_list_all.append([name,'','',''])
    return name_list_all, name_list_highmatch

def save_name_list_csv(name_list_all, name_list_highmatch, file_name):
    df_1 = pd.DataFrame(name_list_all,columns=['raw entity','name match','URL','similar score'])
    df_1.to_csv('./output/name/'+ file_name + '_name_all.csv', index=False)
    df_2 = pd.DataFrame(name_list_highmatch,columns=['raw entity','name match','URL','similar score'])
    df_2.to_csv('./output/name/'+ file_name + '_name_highmatch.csv', index=False)   
    

def extract_name_all(issues_path, name_file, model_path, embedding_file):
    tf.reset_default_graph()
    out_folder = './output/name/'
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    #name_file = '../data/topic_entity/anti-slavery_names-for-tolstoy.csv'
    df = clean_process_name_file(name_file)
    run_ner(issues_path, model_path, embedding_file)
    ann_files, ann_file_names = collect_ann_files()
    
    for i, file_name in enumerate(ann_file_names):
        results_dedupe = get_name_list_per_issue(ann_files[i])
        name_list_all, name_list_highmatch = match_name_list(results_dedupe, df)
        save_name_list_csv(name_list_all, name_list_highmatch, file_name)
 


def find_entity_spacy(data_path):
    with open(data_path) as f:
        data = " ".join(line.strip() for line in f)
    #nlp = spacy.load("en_core_web_sm")
    nlp = en_core_web_sm.load()
    doc = nlp(data)
    entity = [(X.text, X.label_) for X in doc.ents]
    return entity


def export_entity_csv(data_path, csv_dir):
    """Find entity and returns a sublist of [text, label]"""
    data = find_entity_spacy(data_path)
    with open(csv_dir + '/entity.csv','w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['words','entity'])
        for row in data:
            csv_out.writerow(row)
