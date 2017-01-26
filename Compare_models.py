# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 17:38:35 2017

@author: kezhili
"""
from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dropout  
from keras.layers.recurrent import LSTM
import pandas as pd  
import time
import h5py
import numpy as np
import random
#import matplotlib.pyplot as plt
import os
import sys
from matplotlib import pyplot as plt

model = Sequential()  
model.add(LSTM(7, 260, return_sequences=True))  
model.add(LSTM(260, 260, return_sequences=True))  
model.add(Dropout(0.2))  
model.add(LSTM(260, 260, return_sequences=True))  
model.add(Dropout(0.2))  
model.add(LSTM(260, 260, return_sequences=False))  
model.add(Dropout(0.2))   
model.add(Dense(260, 7))  
model.add(Activation("linear"))  
model.compile(loss="mean_squared_error", optimizer="rmsprop")  

# unc-8
stain = "unc-8"  
model_name = 'C:/Users/kezhili/Documents/Python Scripts/data/FromAWS/'+stain+'/multiFile_'+stain+'_7-260-260-260-260-7_600ep.h5'

model.load_weights(model_name)  

weights_unc8 = model.get_weights()

plt.imshow(weights_unc8[1], interpolation='nearest')
plt.show(weights_unc8[1])

# unc-9
stain = "unc-9"  
model_name = 'C:/Users/kezhili/Documents/Python Scripts/data/FromAWS/'+stain+'/multiFile_'+stain+'_7-260-260-260-260-7_600ep.h5'

model.load_weights(model_name)  

weights_unc9 = model.get_weights()

plt.imshow(weights_unc9[1], interpolation='nearest')
plt.show(weights_unc9[1])