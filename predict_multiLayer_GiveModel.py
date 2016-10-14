# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:10:19 2016

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
#
#model = Sequential()  
#model.add(LSTM(7, 200, return_sequences=True))  
#model.add(LSTM(200, 200, return_sequences=True))  
#model.add(Dropout(0.2)) 
#model.add(LSTM(200, 200, return_sequences=True))  
#model.add(Dropout(0.2)) 
#model.add(LSTM(200, 200, return_sequences=False))  
#model.add(Dropout(0.2))   
#model.add(Dense(200, 7))  
#model.add(Activation("linear"))  
#model.compile(loss="mean_squared_error", optimizer="rmsprop")  


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
root = 'Z:/DLWeights/nas207-1/experimentBackup/from pc207-7/!worm_videos/copied_from_pc207-8/Andre/08-03-11/'
#root = 'Z:\DLWeights\nas207-1\experimentBackup\from pc207-7\!worm_videos\copied_from_pc207-8\Andre\03-03-11\'
for path, subdirs, files in os.walk(root):
    for name in files:
        print os.path.join(path, name)   
        file_no = file_no + 1
        
        with h5py.File(os.path.join(path, name), 'r') as fid:
            eig_coef = fid['/eig_coef'][:]  
        
        
        columns = ['a', 'b','c','d','e','f','g']   
        data = pd.DataFrame(eig_coef, columns =  columns)  

        (X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data
        del data
        
        
        ske_arti_step = 0;
        # save model weights
        model.load_weights('C:/Users/kezhili/Documents/Python Scripts/data/FromAWS/test_res_2809_2016/muiltiFile_1000epoch/multiFile_weights.h5')                    

        sentence = X_test[0,:,:]
        eig_generated1 = sentence[-1,]
        
        x_prev = np.zeros(sentence.shape)
        x_prev[1:,] = sentence[0:-1,]
        next_ske = sentence[-1,:]
        
        for ii in range(9000):
            if ii % 50 == 1:
                print "loop =  %d / 9000 ." % ii
            
            x_now = np.zeros((1,sentence.shape[0],sentence.shape[1]))
            #x_now[0:-2,] = x_prev[1:,]
            x_now[0,] =  np.concatenate((x_prev[1:,],[next_ske.T])) 
        
            next_ske = model.predict(x_now, verbose=0)[0]
            
            # add noise to next_ske
            if ske_arti_step == 0:
                if random.random()>0.80:
                    ske_Arti_mag =  (random.random()-0.5)*1.2
                    next_ske[2] = next_ske[2] + ske_Arti_mag
                    ske_arti_step = 1
            else:
                    next_ske[2] = next_ske[2] + ske_Arti_mag*0.8
                    ske_arti_step = 0                    
                
                
            eig_generated1 = np.vstack((eig_generated1, next_ske))
        
            x_prev = np.copy(x_now[0,])
            
            
        name_generated = str(file_no)+"_generated_long_1K_Lnoise.csv"         
        np.savetxt(os.path.join(root, name_generated), eig_generated1, delimiter=",")   
    #    name_y_test = str(file_no)+"_y_test.csv"
    #    np.savetxt(os.path.join(root, name_y_test), y_test, delimiter=",")   
        
        del sentence,eig_generated1,x_prev,next_ske,x_now

