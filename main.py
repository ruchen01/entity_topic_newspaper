#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:37:23 2019

@author: Ru
"""
from __future__ import print_function
import argparse
#import os
import sys
#from src.load import Newspapers
from src.pipeline import test_pipeline
from src.pipeline import major_pipeline

def parse_arguments(arguments = None):
    """Parse the entity_topic_newspaper arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, dest='img_path',
                        help='', default = './data/img')
    parser.add_argument('--topic_file', type=str, dest='topic_file',
                        help='', default = './data/topic_entity/topics.csv')
    parser.add_argument('--entity_file', type=str, dest='entity_file',
                        help='', default = './data/topic_entity/names.csv')
    parser.add_argument('--model_dir', type=str, dest='model_dir',
                        help='', default = './model')
    parser.add_argument('--model_bert_dir', type=str, dest='model_bert_dir',
                        help='', default = './model/uncased_L-12_H-768_A-12')
    parser.add_argument('--model_conll_dir', type=str, dest='model_conll_dir',
                        help='', default = './model/conll_2003_en')
    parser.add_argument('--glove_file', type=str, dest='glove_file',
                        help='', default = './model/glove.6B.100d.txt')
    parser.add_argument('--output_dir', type=str, dest='output_dir',
                        help='', default = './output')
    parser.add_argument('--choice', type=int, dest='choice',
                        help='', default = None)
    try:
        arguments = parser.parse_args(args=arguments)
    except:
        parser.print_help()
        sys.exit(0)
    arguments = vars(arguments)
    return {k:v for k, v in arguments.items() if v is not None}

def main(argv=sys.argv):
    """entity_topic_newspaper main method"""
    arguments = parse_arguments(argv[1:])
    img_dir = arguments['img_path']
    output_text_dir = arguments['output_dir'] + '/text'
    model_bert_dir = arguments['model_bert_dir']
    topic_file = arguments['topic_file']
    name_file = arguments['entity_file']
    model_conll_dir = arguments['model_conll_dir']
    embedding_file = arguments['glove_file']
    # choice of S3 or local directory, by default it's local
    if arguments['choice'] == 0:
        print('train')
        major_pipeline(img_dir, output_text_dir, model_bert_dir, topic_file, name_file, model_conll_dir, 
                       embedding_file)
    elif arguments['choice'] == 1:
        print('test')
        test_pipeline(output_text_dir)
    #file_name = '../13_01_000001.tif'
    #newspapers = Newspapers()
    #newspapers.open_img(file_name)
    #text = newspapers.ocr_img_to_text()
    #print(text)
    #newspapers.test()

if __name__ == "__main__":
    main()
