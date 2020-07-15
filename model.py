# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 17:06:15 2020

@author: Ankur
"""

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
model = load_model( 'rice_leaf.h5')

img = image.load_img('train/Brown spot/DSC_0_524.jpeg',target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
img_data = preprocess_input(x)
classes = model.predict(img_data)

print(classes)