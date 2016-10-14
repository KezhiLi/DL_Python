# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:17:39 2016

@author: kezhili
"""

from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dropout  
from keras.layers.recurrent import LSTM

model = Sequential()  
model.add(LSTM(6, 200, return_sequences=True))  
model.add(LSTM(200, 200, return_sequences=True))  
model.add(Dropout(0.2)) 
#model.add(LSTM(200, 200, return_sequences=True))  
#model.add(Dropout(0.2)) 
model.add(LSTM(200, 200, return_sequences=False))  
model.add(Dropout(0.2))   
model.add(Dense(200, 6))  
model.add(Activation("linear"))  
model.compile(loss="mean_squared_error", optimizer="rmsprop")  


import pandas as pd  
import time
import h5py
import numpy as np
import sys

#file_name = sys.argv[1]
#
#with h5py.File(file_name+'_eig.hdf5', 'r') as fid:
#    eig_coef = fid['/eig_coef'][:]   

file_name = sys.argv[1]

with h5py.File('Z:/DLWeights/nas207-1/experimentBackup/from pc207-7/!worm_videos/copied_from_pc207-8/Andre/03-03-11/247 JU438 on food L_2011_03_03__11_18___3___1_eig.hdf5', 'r') as fid:
    eig_coef = fid['/eig_coef'][:]    
   
columns = ['a', 'b','c','d','e','f']    
data = pd.DataFrame(eig_coef, columns =  columns)    

def _load_data(data, n_prev = 50):  
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
model.fit(X_train, y_train, batch_size=450, nb_epoch=400, validation_split=0.05)  
        
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
np.savetxt(file_name+"_eig(4hid).csv", eig_generated1, delimiter=",")              
# save model weights
model.save_weights(file_name+ '_weights(4hid).h5')    
