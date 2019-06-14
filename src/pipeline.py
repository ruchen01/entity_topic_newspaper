import sys

import extract_entity
import extract_topic



def main(argv):
    #data_path = argv[1]
    text_file = '../data/preprocessed/13_01_000002/1.txt'
    out_dir = '../out'
    text_dir = '../data/preprocessed/13_01_000002'

    #add OCR part here
    extract_entity.export_entity_csv(text_file, out_dir)
    extract_topic.print_topic(text_dir, out_dir)


if __name__ == '__main__':
	main(sys.argv)




