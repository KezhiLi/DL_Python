# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 17:38:35 2017

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
import h5py
import numpy as np

import theano.tensor as T
from keras.utils.theano_utils import alloc_zeros_matrix

class LSTM_f(LSTM):
    def __init__(self, *args, **kwargs):
        super(LSTM_f, self).__init__(*args, **kwargs)
    
    def get_output_f(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o

        [outputs, memories], updates = theano.scan(
            self._step,
            sequences=[xi, xf, xo, xc, padded_mask],
            outputs_info=[
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
            ],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2)), memories.dimshuffle((1, 0, 2))
        return outputs[-1], memories[-1]
    

model = Sequential()  
L0 = LSTM_f(7, 260, return_sequences=True)
model.add(L0)
L1 = LSTM_f(260, 260, return_sequences=True)
model.add(L1)  
model.add(Dropout(0.2))  
L2 = LSTM_f(260, 260, return_sequences=True)
model.add(L2)  
model.add(Dropout(0.2))  
L3 = LSTM_f(260, 260, return_sequences=False)
model.add(L3)  
model.add(Dropout(0.2))   
model.add(Dense(260, 7))
A_out = Activation("linear")
model.add(A_out)  
model.compile(loss="mean_squared_error", optimizer="rmsprop")  ######

stain =  "wild-isolate"  

root = 'Z:\DLWeights\eig_catagory_Straits\wild-isolate'
for path, dirs, files in os.walk(root):
    for name in files:
        if name.endswith(("_eig.hdf5")):
        #    name = '135 CB4852 on food R_2011_03_30__12_52_35___7___5_eig'
            with h5py.File('Z:/DLWeights/eig_catagory_Straits/'+stain + '/'+name,'r') as fid:
                eig_coef = fid['/eig_coef'][:]
                eig_vec  = fid['/eig_vec'][:]
                len_vec  = fid['/len_vec'][:]
                mean_angle_vec  = fid['/mean_angle_vec'][:]
            #    nan_idx  = fid['/nan_idx'][:]
            
               
            columns = ['a', 'b','c','d','e','f','g']    
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
            
            def train_test_split(df, test_size=0.9):  
                """
                This just splits data to training and testing parts
                """
                if test_size==1:
                    X_train, y_train = [],[]
                    X_test, y_test = _load_data(df.iloc[:])
                else:    
                    ntrn = int(round(len(df) * (1 - test_size)))
              
                    X_train, y_train = _load_data(df.iloc[0:ntrn])
                    X_test, y_test = _load_data(df.iloc[ntrn:])
            
                return (X_train, y_train), (X_test, y_test)
                
            t0 = time.time()    
            
            (X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data
            
            # unc-8
            model_name = 'C:/Users/kezhili/Documents/Python Scripts/data/FromAWS/'+stain+'/multiFile_'+stain+'_7-260-260-260-260-7_600ep.h5'
            model.load_weights(model_name)  
            
            
            predicted = model.predict(X_test)  
            rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
            
            model_L0_func = theano.function([model.layers[0].input], model.layers[0].get_output(train=False), allow_input_downcast=True)
            res_L0 = model_L0_func(X_test)
            
            
            def get_c(model, layer, X_batch):
                get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output_f(train=False), allow_input_downcast=True)
                res_L, c_L = get_activations(X_batch) # same result as above
                return res_L, c_L
                
            # res_L and c_L have size [8000,50,260]    
            res_L0,c_L0 = get_c(model, 0, X_test)  
            res_L0_col, c_L0_col = np.squeeze(res_L0[:,-1,:]), np.squeeze(c_L0[:,-1,:])
            del res_L0,c_L0
            
            res_L1,c_L1 = get_c(model, 1, X_test)
            res_L1_col, c_L1_col = np.squeeze(res_L1[:,-1,:]), np.squeeze(c_L1[:,-1,:])
            del res_L1,c_L1
            
            res_L3,c_L3 = get_c(model, 3, X_test)
            res_L3_col, c_L3_col = np.squeeze(res_L3[:,-1,:]), np.squeeze(c_L3[:,-1,:])
            del res_L3,c_L3
            
            res_L5_col,c_L5_col = get_c(model, 5, X_test) 
            
            
            data_name = 'interm_'+name[:-5]+'_7-260-260-260-260-7_600ep.hdf5'
            
            h5f = h5py.File('Z:/DLWeights/eig_catagory_Straits/'+ stain +'/'+data_name, 'w')
            h5f.create_dataset('y_test', data=y_test)
            h5f.create_dataset('c_L0_col', data=c_L0_col)
            h5f.create_dataset('c_L1_col', data=c_L1_col)
            h5f.create_dataset('c_L3_col', data=c_L3_col)
            h5f.create_dataset('c_L5_col', data=c_L5_col)
            
            h5f.create_dataset('res_L0_col', data=res_L0_col)
            h5f.create_dataset('res_L1_col', data=res_L1_col)
            h5f.create_dataset('res_L3_col', data=res_L3_col)
            h5f.create_dataset('res_L5_col', data=res_L5_col)
            
            h5f.create_dataset('eig_vec', data=eig_vec)
            h5f.create_dataset('len_vec', data=len_vec)
            h5f.create_dataset('mean_angle_vec', data=mean_angle_vec)
            #h5f.create_dataset('nan_idx', data=nan_idx)
            
            h5f.close()                                                 