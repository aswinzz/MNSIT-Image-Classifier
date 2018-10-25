import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from keras.models import load_model
from PIL import Image
import sys
from keras import backend as K


K.clear_session()
model = load_model('model.h5')
pic = Image.open("images/0_a.png")
Pic = np.array(pic)
x = Pic.reshape((1,)+Pic.shape+(1,))
val = model.predict(x)
print(val[0].argmax(axis=0))
