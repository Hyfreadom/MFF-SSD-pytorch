import os
import numpy as np
from PIL import Image
from ssd import SSD

'''
人脸编码网络
'''
ssd_face = SSD()
pic_names = os.listdir("source_img")
image_paths = []
names = []
face_encodings = []
for pic_name in pic_names:
    names.append(pic_name[:-4])
print(names)
for name in names:
    image_path ='source_img/'+name+'.jpg'
    image = Image.open(image_path)
    face_encodings.append(ssd_face.encode_image(image,names))
np.save("model_data/{backbone}_face_encoding.npy".format(backbone='mobilenet'),face_encodings)
np.save("model_data/{backbone}_names.npy".format(backbone='mobilenet'),names)
