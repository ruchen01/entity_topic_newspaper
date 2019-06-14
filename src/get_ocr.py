import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import boto3
import io
from PIL import Image

s3 = boto3.resource('s3', region_name='us-west-2')
bucket = s3.Bucket('newspaper-ru-insight')
object = bucket.Object('13_01_000001.tif')

file_stream = io.BytesIO()
object.download_fileobj(file_stream)
print(file_stream)
image = Image.open(file_stream)


#img = mpimg.imread(file_stream)
