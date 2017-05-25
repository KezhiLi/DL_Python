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

for jj in range(9):
    
    # whatever
    if jj == 0:
        stain =  'CB4852'
    elif jj == 1:
        stain =  'ED3017'    
    elif jj == 2:
        stain =  'ED3052'  
    elif jj == 3:
        stain =  'JU298'  
    elif jj == 4:
        stain =  'JU345'  
    elif jj == 5:
        stain =  'JU438'  
    elif jj == 6:
        stain =  'LSJ1'  
    elif jj == 7:
        stain =  'MY16'   
    elif jj == 8:
        stain =  'ser-6'      
    else:
        continue
    
    print stain
    
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
    
    file_no = 0
    root = './DLWeights/eig_catagory_Straits/' + stain + '/'
    #root = 'Z:\DLWeights\nas207-1\experimentBackup\from pc207-7\!worm_videos\copied_from_pc207-8\Andre\03-03-11\'
    for path, subdirs, files in os.walk(root):
        for name in files: 
            if name.endswith("eig.hdf5"):
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
            # save model weights
    model.save_weights('./DLWeights/eig_catagory_Straits/'+stain+'/multiFile_'+stain +'_7-260-260-260-260-7_600ep.h5')                    
 
    del model,  X_train, y_train