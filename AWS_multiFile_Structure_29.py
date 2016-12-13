# -*- coding: utf-8 -*-
"""
Created on Mon May 23 19:13:57 2016

@author: kezhili
"""
#from keras.models import Model
from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dropout  
from keras.layers.recurrent import LSTM
import pandas as pd  
import time
import h5py
import numpy as np
#from random import random
#import matplotlib.pyplot as plt
import os

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

file_no = 0
root = './DLWeights/eig_catagory_Straits/tdc-1/'
#root = 'Z:\DLWeights\nas207-1\experimentBackup\from pc207-7\!worm_videos\copied_from_pc207-8\Andre\03-03-11\'
for path, subdirs, files in os.walk(root):
    for name in files: 
        if name.endswith(".hdf5"):
            print os.path.join(path, name)   
            file_no = file_no + 1
            
            with h5py.File(os.path.join(path, name), 'r') as fid:
                eig_coef = fid['/eig_coef'][:]  
            
            
            columns = ['a', 'b','c','d','e','f','g']   
            data = pd.DataFrame(eig_coef, columns =  columns)  
    
            (X_train_curr, y_train_curr), (X_test, y_test) = train_test_split(data)  # retrieve data
            del data
            
            if file_no == 1:
                X_train = X_train_curr
                y_train = y_train_curr
            else:    
                X_train = np.concatenate((X_train,X_train_curr), axis=0)
                y_train = np.concatenate((y_train,y_train_curr), axis=0)
            del X_train_curr,y_train_curr   
            
    
# and now train the model
# batch_size should be appropriate to your memory size
# number of epochs should be higher for real world problems
model.fit(X_train, y_train, batch_size=450, nb_epoch=600, validation_split=0.05)  

#predicted = model.predict(X_test)  
#rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
#
## and maybe plot it
##pd.DataFrame(predicted[:50]).plot()  
##pd.DataFrame(y_test[:50]).plot()  
#
## save to csv
#np.savetxt("./DLWeights/nas207-1/experimentBackup/from pc207-7/!worm_videos/copied_from_pc207-8/Andre/03-03-11/predicted.csv", predicted, delimiter=",")
#np.savetxt("./DLWeights/nas207-1/experimentBackup/from pc207-7/!worm_videos/copied_from_pc207-8/Andre/03-03-11/y_test2.csv", y_test, delimiter=",")
#

        
#sentence = X_test[0,:,:]
#eig_generated1 = sentence[-1,]
#
#x_prev = np.zeros(sentence.shape)
#x_prev[1:,] = sentence[0:-1,]
#next_ske = sentence[-1,:]
#
#for ii in range(400):
#    if ii % 50 == 1:
#        print "loop =  %d / 400 ." % ii
#    
#    x_now = np.zeros((1,sentence.shape[0],sentence.shape[1]))
#    #x_now[0:-2,] = x_prev[1:,]
#    x_now[0,] =  np.concatenate((x_prev[1:,],[next_ske.T])) 
#
#    next_ske = model.predict(x_now, verbose=0)[0]
#    eig_generated1 = np.vstack((eig_generated1, next_ske))
#
#    x_prev = np.copy(x_now[0,])
#    
#print time.time() - t0 
#
#name_generated = "generated.csv"
              
# save model weights
model.save_weights('./DLWeights/eig_catagory_Straits/tdc-1/multiFile_tdc-1_7-260-260-260-260-7_600ep.h5')                    
## save to csv
#np.savetxt(os.path.join(root, name_generated), eig_generated1, delimiter=",")     
