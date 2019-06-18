import boto3
import io
from PIL import Image
import pytesseract


def img2text_s3(bucket_name, text_folder='text', img_folder='img')
"""For given img folder in s3, generate OCR'd text into the text folder"""
    #bucket_name = 'newspaper-ru-insight'
    s3 = boto3.resource('s3', region_name='us-west-2')
    bucket = s3.Bucket(bucket_name)
    max_cnt = 200
    cnt = 0
    for obj in bucket.objects.filter(Delimiter='/', Prefix=img_folder+'/'):
      print(obj.key)
      img_file = obj.key
      object=bucket.Object(img_file)
      if cnt < max_cnt:
        object=bucket.Object(obj.key)
        file_stream = io.BytesIO()
        object.download_fileobj(file_stream)
        #print(file_stream)
        image = Image.open(file_stream)
        text = pytesseract.image_to_string(image)
        text_file = text_folder + img_file[3:-3] +'txt'
        object=s3.Object(bucket_name, text_file)
        object.put(Body=text)
        cnt += 1
      else:
        break
    
