#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:25:41 2019

@author: Ru
"""
#import pytesseract
from PIL import Image
import nltk
import re
import pytesseract

class Newspapers(object):
    """Load multiple newspaper images and write out text files
    The text files are got by OCR 
    """
    def __init__(self):
        #self.image = None
        self.texts = []
            
    def open_img(self, file_name):
        self.image = Image.open(file_name)
        return self.image
        
    def img_to_text(self, location_type='local'):
        
            
    
        self.text = pytesseract.image_to_string(self.image)
        return self.text
        
    def test(self):
        print('newspaper test test')


class Newspaper(object):
    """Perform actions on newspaper, text processing and cleaning.
    Major methods:
    img_to_text() --- OCR image file into text.
    issue2articles() --- divide issues by articles, important for topic extraction. 
    
    """
    def __init__(self):
        self.image = None
        self.text = None
        self.name = None
            
    def open_img(self, img_file):
        """ 
        """
        self.image = Image.open(img_file)
        return self.image
        
    def img_to_text(self):
        """
        """
        self.text = pytesseract.image_to_string(self.image)
        return self.text
    
    def save_text(self,text_file):
        with open(text_file, 'w') as f:
            f.write(self.text)
        
    def tex_to_articles(self):
        """
        """
        print('newspaper test test')
        
    def load_text(self, text_file):
        with open(text_file) as f:
            self.text = f.readlines()        
        return self.text
        
        
    def issue2articles(self):
        """Break each issue into articles based on the style of titles
        All the titles of the articles are in capital letters.
        """
        data = self.text
        divide_li = []
        for i, line in enumerate(data):
            if i<len(data):
                tok = nltk.word_tokenize(line)
                tok_cap = []
                for word in tok:
                    if len(word)==1:
                        cap_word = re.findall('([A-Z])', word)
                    else:
                        cap_word = re.findall('([A-Z]+(?:(?!\s?[A-Z][a-z])\s?[A-Z])+)', word)
                    if cap_word != []:
                        tok_cap.append(cap_word[0]) 
                if tok_cap != [] and len(tok)-len(tok_cap)<=1 and len(tok)>2:
                    divide_li.append(i)
        articles = []
        for i,ind in enumerate(divide_li):
            if i+1<= len(divide_li)-1:
                if divide_li[i+1]-ind>5:
                    articles.append(data[ind:divide_li[i+1]])
            else:
                articles.append(data[ind:])
        # The first article per issue is always the headline and therefore is dropped
        articles.pop(0);
        articles_list=[]
        for article in articles:
            data = " ".join(line.strip() for line in article) 
            articles_list.append(data)
        return articles_list
    
    def articles2sentences(self, articles_list):
        """Break each article into sentences for each issue. issue_sentence is a flat list 
        of sentences and issue_article_sentence is nested list. """
        issue_sentence = []
        issue_article_sentence = []
        for article in articles_list:
            tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            sentences = tokenizer.tokenize(article)
            article_sentence=[]
            for sentence in sentences:
                if len(sentence)>5:
                    issue_sentence.append(sentence)
                    article_sentence.append(sentence)
            issue_article_sentence.append(article_sentence)
        return issue_sentence, issue_article_sentence
    
    def find_divide_article_line_number(self, issue_article_sentence):
        """Find the line number which divide each issue into articles """
        divide_num = 0
        divide_list = []
        for idx, article in enumerate(issue_article_sentence):
            divide_num += len(article)
            divide_list.append(divide_num)
        return divide_list
    
    



    
    #return img_dir
        
#imgs_to_texts_save_local()
    
#text_file = '/Users/Ru/Documents/python/DL_projs/insight_AI/test_2/output_short/13_01_000006.txt'
#x = Newspaper()
#x.load_text(text_file)
#article_list = x.issue2articles()
##print(article_list)
#issue_sentence, issue_article_sentence=x.articles2sentences(article_list)
#divide_list = x.find_divide_article_line_number(issue_article_sentence)

#print(x.texts)
#print(x.images)
#x.ocr_img()
#x.load_img_from_local_dir()
#print(x.texts)
