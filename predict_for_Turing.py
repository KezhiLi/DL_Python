# -*- coding: utf-8 -*-
"""
Created on March 27  2017

@author: kezhili
"""

from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import theano
import pandas as pd  
import time
import h5py
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd  
import time

# as the first layer in a Sequential model

model = Sequential()  
L0 = LSTM(7, 260, return_sequences=True)
model.add(L0)
L1 = LSTM(260, 260, return_sequences=True)
model.add(L1)  
model.add(Dropout(0.2))  
L2 = LSTM(260, 260, return_sequences=True)
model.add(L2)  
model.add(Dropout(0.2))  
L3 = LSTM(260, 260, return_sequences=False)
model.add(L3)  
model.add(Dropout(0.2))   
model.add(Dense(260, 7))
A_out = Activation("linear")
model.add(A_out)  
model.compile(loss="mean_squared_error", optimizer="rmsprop")  ######

def _load_data(data, n_prev):  
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

def train_test_split(df, test_size, n_prev):  
    """
    This just splits data to training and testing parts
    """
    ntrn = int(round(len(df) * (1 - test_size)))

    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)

    return (X_train, y_train), (X_test, y_test)

file_no = 0

root = 'Z:/DLWeights/nas207-1/experimentBackup/'
root_simulate = 'Z:/Ken_Samples/simulated/'
for path, dirs, files in os.walk(root):
    for name in files:
        try:
            if name.endswith(("_eig.hdf5")) and ('on food' in name):
                # whatever
                if "N2" in name:
                    stain =  'N2'
                elif "unc-8" in name:
                    stain =  'unc-8'    
                elif "ser-6" in name:
                    stain =  'ser-6'  
                elif "tdc-1" in name:
                    stain =  'tdc-1'  
                elif "tbh-1" in name:
                    stain =  'tbh-1'  
                elif "cb4856" in name:
                    stain =  'cb4856'  
                elif "unc-9" in name:
                    stain =  'unc-9'  
                elif "trp-4" in name:
                    stain =  'trp-4'   
                elif ("MY" in name) or ("LS" in name) or ("JU" in name) or ("ED" in name) or ("CB" in name) or ("AQ" in name):
                    stain =  'wild-isolate'  
                else:
                    continue
                
                print os.path.join(path, name) 
                                
                model_name = 'C:/Users/kezhili/Documents/Python Scripts/data/FromAWS/'+stain+'/multiFile_'+stain+'_7-260-260-260-260-7_600ep.h5' 
                model.load_weights(model_name) 
                
                with h5py.File(os.path.join(path, name), 'r') as fid:
                    eig_coef = fid['/eig_coef'][:]  
               
                columns = ['a','b','c','d','e','f','g']   
                data = pd.DataFrame(eig_coef, columns =  columns)  
    
                n_prev = 50    
                len_data = len(data)
                if len_data < n_prev+1:
                    n_prev = len_data - 1
                    start_fraction = 1
                    start_idx = 0
                else:
                    start_fraction = random.random()*(float(len_data-n_prev-2)/float(len_data))
                    start_idx = int(round(len_data * (1 - start_fraction)))
                     
                (X_train, y_train), (X_test, y_test) = train_test_split(data,start_fraction,n_prev)  # retrieve data
                del data
        
                sentence = X_test[0,:,:]
                eig_generated1 = sentence[-1,]
                
                x_prev = np.zeros(sentence.shape)
                x_prev[1:,] = sentence[0:-1,]
                next_ske = sentence[-1,:]
                
                name_suf = "_samp_"+str(start_idx)+".csv"
            
                for ii in range(700):
                    if ii % 50 == 1:
                        print "loop =  %d / 700 ." % ii
                    
                    x_now = np.zeros((1,sentence.shape[0],sentence.shape[1]))
                    #x_now[0:-2,] = x_prev[1:,]
                    x_now[0,] =  np.concatenate((x_prev[1:,],[next_ske.T])) 
                
                    next_ske = model.predict(x_now, verbose=0)[0]
                 
                    eig_generated1 = np.vstack((eig_generated1, next_ske))
                
                    x_prev = np.copy(x_now[0,])
                    
                    
                name_generated = name[:-9]+  name_suf   
                path_generated = root_simulate + path[13:]
                obj_csv = os.path.join(path_generated, name_generated).replace("\\",'/')
                obj_path = path_generated.replace("\\",'/')
                if not os.path.exists(obj_path):
                    os.makedirs(obj_path)
                np.savetxt(obj_csv, eig_generated1, delimiter=",")   
                
                del sentence,eig_generated1,x_prev,next_ske,x_now
                file_no = file_no + 1
        except:
            pass