# Semantic Search
![Preview](https://github.com/ruchen01/entity_topic_newspaper/assets/header.png)

This repository contains a barebones implementation of a topic and name entity extraction engine.
The implementation is based on leveraging pre-trained models from BERT for topic modeling and 
GloVe for name entity.

It allows you to:
- Get OCR'd text from an input scanned image of newspaper.
- Extract relevant topics of newspaper and match them with Library of Congress subject listing. 
- Extract important person's names of newspaper and match with against Library of Congress name listing.


## Setup
Clone the repository locally and create a virtual environment (conda example below):
```
conda create -n librarybot python=3.6 -y
source activate librarybot
cd entity_topic_newspaper
pip install -r requirements.txt
```

Download a pre-trained BERT model, the example download provided below is BERT-Base, Uncased. 
Download GloVe vectors (vector size 300):
```
mkdir model
curl -LO https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
curl -LO http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

```

Two example images are already included at data/img folder, so you can start test run immediately.





## Creating a custom dataset
Image dataset must be of the format below if you would like to import your own:
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
