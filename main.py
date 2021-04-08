import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Activation,Dense,Dropout,Flatten
from tensorflow.python.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import path
from PIL import Image
model = keras.models.load_model('my_model.h5')
put = input()
n = os.path.join(os.getcwd(), os.path.normpath(put))
image = Image.open(n)
size = (150, 150)
image = image.resize(size)
img = img_to_array(image)
img = np.expand_dims(img,axis=0)
img /= 255.
yhat = model.predict(img)
predicted_class = model.predict_classes(img)[0]
if predicted_class == 1:
    print('Мухомор')
else:
    print('Боровик')




