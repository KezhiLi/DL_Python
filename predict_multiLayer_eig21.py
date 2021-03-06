# -*- coding: utf-8 -*-
"""
Created on Mon May 23 19:13:57 2016

@author: kezhili
"""

from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dropout  
from keras.layers.recurrent import LSTM

## single hidden layer
#in_out_neurons = 5  
#hidden_neurons = 1000
#model = Sequential()  
#model.add(LSTM(in_out_neurons, hidden_neurons, return_sequences=False))  
#model.add(Dense(hidden_neurons, in_out_neurons))  
#model.add(Activation("linear"))  
#model.compile(loss="mean_squared_error", optimizer="rmsprop")  

model = Sequential()  
#model.add(LSTM(6, 500, return_sequences=True))  
#model.add(LSTM(500, 500, return_sequences=False))  
model.add(LSTM(6, 300, return_sequences=False))  
model.add(Dropout(0.2))  
model.add(Dense(300, 6))  
model.add(Activation("linear"))  
model.compile(loss="mean_squared_error", optimizer="rmsprop")  

import pandas as pd  
import time
#from random import random
#import matplotlib.pyplot as plt

#import scipy.io
#mat = scipy.io.loadmat('angle_vec.mat')

import h5py
import numpy as np
with h5py.File('./data/eig_para_full21(247JU438).hdf5', 'r') as fid:
    eig_coef = fid['/eig_coef'][:]     
   
columns = ['a', 'b','c','d','e','f']    
data = pd.DataFrame(eig_coef, columns =  columns)    

def _load_data(data, n_prev = 30):  
    """
    data should be pd.DataFrame()
    """
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1):  
    """
    This just splits data to training and testing parts
    """
    ntrn = int(round(len(df) * (1 - test_size)))

    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])

    return (X_train, y_train), (X_test, y_test)
    
t0 = time.time()    

(X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data

# and now train the model
# batch_size should be appropriate to your memory size
# number of epochs should be higher for real world problems
model.fit(X_train, y_train, batch_size=450, nb_epoch=300, validation_split=0.05)  

predicted = model.predict(X_test)  
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))

# and maybe plot it
pd.DataFrame(predicted[:50]).plot()  
pd.DataFrame(y_test[:50]).plot()  

## save to csv
#np.savetxt("./data/eig_MulLayer_predicted_full21(247JU438)-1hid.csv", predicted, delimiter=",")
#np.savetxt("./data/eig_MulLayer_y_test_full21(247JU438)-1hid.csv", y_test, delimiter=",")



### pure predict
#
#def sample(a, temperature=1.0):
#    # helper function to sample an index from a probability array
#    a = np.log(a) / temperature
#    a = np.exp(a) / np.sum(np.exp(a))
#    return np.argmax(np.random.multinomial(1, a, 1))
    
    
        
sentence = X_test[0,:,:]
eig_generated1 = sentence[-1,]

x_prev = np.zeros(sentence.shape)
x_prev[1:,] = sentence[0:-1,]
next_ske = sentence[-1,:]

for ii in range(400):
    if ii % 50 == 1:
        print "loop =  %d / 400 ." % ii
    
    x_now = np.zeros((1,sentence.shape[0],sentence.shape[1]))
    #x_now[0:-2,] = x_prev[1:,]
    x_now[0,] =  np.concatenate((x_prev[1:,],[next_ske.T])) 

    next_ske = model.predict(x_now, verbose=0)[0]
    eig_generated1 = np.vstack((eig_generated1, next_ske))

    x_prev = np.copy(x_now[0,])
                
# save to csv
#np.savetxt("./data/eig_MulLayer_generated_full21(247JU438)-1hid.csv", eig_generated1, delimiter=",")     

print time.time() - t0               

# save model weights
#model.save_weights('./data/eig_MulLayer_generated_full21(247JU438)-1hid.h5')    
