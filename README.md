# LibraryBot - Topic and Entity Extraction for Historic Newspapers
<img src="assets/header.png" >

## What is it?

This repository contains a barebones implementation of a topic and name entity extraction engine.
The implementation is based on leveraging pre-trained models from BERT combining with TFIDF for topic modeling and 
GloVe for name entity extraction.

It allows you to:
- Get OCR'd text from an input scanned image of newspaper.
- Extract relevant topics of newspaper and match them with Library of Congress subject listing. 
- Extract important person's names of newspaper and match them with Library of Congress name listing.

BERT is a masked language model developed by [Google](https://github.com/google-research/bert). It contains two steps in the framework: pre-training and fine-tuning. During the pre-training, the model is trained on unlabeled BooksCorpus and English Wikipedia text. For fine-tuning, the BERT model is initialized with the pre-trained parameters with them fine-tuned for the downstream tasks. In the [paper](https://arxiv.org/abs/1810.04805), it is shown that the feature-based approach with BERT, where fixed features are extracted from the pre-trained model, is only slightly behind fine-tuning the entire model. One can then perform sentence encoding using [BERT](https://github.com/hanxiao/bert-as-service) , which goal is to represent a variable length sentence into a fixed length vector, e.g. `hello world` to `[0.1, 0.3, 0.9]`.   

## Setup
Clone the repository locally and create a virtual environment (conda example below):
```
conda create -n librarybot python=3.6 -y
source activate librarybot
cd entity_topic_newspaper
pip install -r requirements.txt
```

Download a pre-trained BERT model, the example download provided below is BERT-Base, Uncased. 
Install Tesseract package, download SpaCy English and GloVe vectors (vector size 300):
```
cd model
curl -LO https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
curl -LO http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
cd ..
python -m spacy download en
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
```

Two example images are already included at data/img folder, so you can start test run immediately.

## Usage
### Running the model with pipeline end to end
Here is an example for running the pipeline end to end with default settings! 
```
python main.py 
```

### Run test 
After running the model, if you want to add additional new scanned images of newspaper:
```
python main.py --test
```

### More configurations
...


## Creating a custom dataset
Data must be of the format below if you would like to import your own:
```
data/
|
|--- img/
|      |-------image_0
|      |-------image_1
|      ...
|
|      |-------image_n
|--- topic_entity/
|      |-------topic_list.csv
|      |-------entity_list.csv
```
Each class name should be one word in the english language, or multiple words separated by '_'.

## Output structure
The revelant output directory are topic, name and text, where topic folder contains the relevant topic list per issue and 
name gives the important person's name list per issue. The text folder contains the OCR'd texts from images and can be reused for other purposes.

```
output/
|
|--- topic/
|      |-------issue_1.txt
|      |-------issue_2.txt
|      ... 
|
|      |-------issue_n.txt
|--- name/
|      |-------issue_1.csv
|      |-------issue_2.csv
|      ...
|      |-------issue_n.csv
|----text/
|      |-------issue_1.txt
|      |-------issue_2.txt
|      ...
|
|      |-------issue_n.txt
|----ner/
```
