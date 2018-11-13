
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
import sys
import os

model = load_model('model.h5')
directory = os.fsencode("./images/")
ans=[]
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    # print(filename)
    file_name="images/"+filename
    pic = Image.open(file_name)
    Pic = np.array(pic)
    x = Pic.reshape((1,)+Pic.shape+(1,))
    val = model.predict(x)
    # print(val[0].argmax(axis=0))
    pred=val[0].argmax(axis=0)
    if(pred not in ans):
        ans.append(pred)
        print(pred)
    else:
        # print(pred," is Already present")
        pass
