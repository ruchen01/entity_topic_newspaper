import boto3
import io
import os
import glob
from src.process import Newspaper


def imgs_to_texts_save_local(img_dir='./data/img', output_dir ='./output/text'):
    """ Save all OCR'd text to local directory    """
    img_files = glob.glob(img_dir + '/*')
    print(img_files)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for file in img_files:
        # dir/xxxxxx.tif
        newspaper_name = file[12:-4]
        #print(newspaper_name)       
        x = Newspaper()
        x.open_img(file)
        x.img_to_text()
        text_file = output_dir + '/' + newspaper_name + '.txt'
        x.save_text(text_file)
           
    
def imgs_to_texts_save_s3(bucket_name, text_folder='text', img_folder='img'):
    """For given img folder in s3, generate OCR'd text into the text folder"""
    #bucket_name = 'newspaper-ru-insight'
    #s3 = boto3.resource('s3', region_name='us-west-2')
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Delimiter='/', Prefix=img_folder+'/'):
        img_file = obj.key
        object=bucket.Object(img_file)
        file_stream = io.BytesIO()
        object.download_fileobj(file_stream)
        x = Newspaper()
        x.open_img(file_stream)
        text = x.img_to_text()
        text_file = text_folder + img_file[3:-3] +'txt'
        object=s3.Object(bucket_name, text_file)
        object.put(Body=text)
  
    
