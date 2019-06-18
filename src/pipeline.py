import sys
import extract_entity
import extract_topic
import get_ocr


def main(argv):
    #data_path = argv[1]
    text_file = '../data/preprocessed/13_01_000002/1.txt'
    out_dir = '../out'
    text_dir = '../data/preprocessed/13_01_000002'
    bucket_name ='newspaper-ru-insight' 
    img2text_s3(bucket_name, text_folder='text', img_folder='img')
    extract_entity.export_entity_csv(text_file, out_dir)
    extract_topic.print_topic(text_dir, out_dir)


if __name__ == '__main__':
	main(sys.argv)




