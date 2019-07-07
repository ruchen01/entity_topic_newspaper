from __future__ import print_function
import argparse
import sys
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
    parser.add_argument('--test', type=int, dest='test',
                        help='', default = False)
    parser.add_argument('--gpu', type=bool, dest='cpu',
                        help='', default = False)
    parser.add_argument('--s3', type=bool, dest='s3',
                        help='', default=False)
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
    if arguments['test'] == False:
        print('Runing the pipeline from end-to-end! ')
        major_pipeline(img_dir, output_text_dir, model_bert_dir, topic_file, name_file, model_conll_dir, 
                       embedding_file)
    else:
        print('Run test.')
        test_pipeline(output_text_dir)

if __name__ == "__main__":
    main()
