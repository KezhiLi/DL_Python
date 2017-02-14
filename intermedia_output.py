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

#model = Sequential()
#model.add(LSTM(input_dim=7,  output_dim=260, return_sequences=True))#input_length=50,
#model.add(LSTM(input_dim=260,output_dim=260, return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(input_dim=260,output_dim=260, return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(input_dim=260,output_dim=260, return_sequences=False))
#model.add(Dropout(0.2))
#model.add(Dense(input_dim=260, output_dim=7))  
#model.add(Activation("linear"))  
#model.compile(loss="mean_squared_error", optimizer="rmsprop")  



#from random import random
#import matplotlib.pyplot as plt

#import scipy.io
#mat = scipy.io.loadmat('angle_vec.mat')

import h5py
import numpy as np

# 'C:/Users/kezhili/Documents/Python Scripts/data/FromAWS/wild-isolate/MY16/800 MY16 on food R_2011_03_17__15_53___3___8_eig.hdf5'

name = 'N2 on food L_2011_02_17__12_51_07___7___7_eig'
with h5py.File('Z:/DLWeights/eig_catagory_Straits/N2/'+name+'.hdf5','r') as fid:
    eig_coef = fid['/eig_coef'][:]
    eig_vec  = fid['/eig_vec'][:]
    len_vec  = fid['/len_vec'][:]
    mean_angle_vec  = fid['/mean_angle_vec'][:]
#with h5py.File('./data/eig_para_full22.hdf5', 'r') as fid:
#    eig_coef = fid['/eig_coef'][:]     
   
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
stain =  'N2'# "wild-isolate"  
model_name = 'C:/Users/kezhili/Documents/Python Scripts/data/FromAWS/'+stain+'/multiFile_'+stain+'_7-260-260-260-260-7_600ep.h5'
model.load_weights(model_name)  

# and now train the model
# batch_size should be appropriate to your memory size
# number of epochs should be higher for real world problems
#model.fit(X_train, y_train, batch_size=450, nb_epoch=500, validation_split=0.05)  

predicted = model.predict(X_test)  
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))

#from keras import backend as K
#get_1st_layer_output = K.function([model.layers[0].input],[model.layers[1].output])
#first_layer_output = get_1st_layer_output([X_test])[0]

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


data_name = 'interm_'+name+'_7-260-260-260-260-7_600ep.hdf5'

h5f = h5py.File('Z:/DLWeights/eig_catagory_Straits/N2/'+data_name, 'w')
h5f.create_dataset('y_test', data=y_test)
h5f.create_dataset('c_L0_col', data=c_L0_col)
h5f.create_dataset('c_L1_col', data=c_L1_col)
h5f.create_dataset('c_L3_col', data=c_L3_col)
h5f.create_dataset('c_L5_col', data=c_L5_col)
h5f.create_dataset('eig_vec', data=eig_vec)
h5f.create_dataset('len_vec', data=len_vec)
h5f.create_dataset('mean_angle_vec', mean_angle_vec)

h5f.close()

#[res_L1,c_L1] = model_ful_L1_func(X_test)
#
#[res_L3,c_L3] = model_ful_L3_func(X_test)
#
#[res_L5,c_L5] = model_ful_L5_func(X_test)

#model_L_func = theano.function([model.layers[0].input], [model.layers[0].get_output(train=False),model.layers[0].get_output(train=False)] allow_input_downcast=True)
#res_L = model_L_func(X_test)

#import theano
#get_activations = theano.function([model.layers[0].input], model.layers[1].output(train=False), allow_input_downcast=True)
#activations = get_activations(X_batch) 

#first_layer_output = keras.layer.get_output_at(node_index)


## and maybe plot it
#pd.DataFrame(predicted[:50]).plot()  
#pd.DataFrame(y_test[:50]).plot()  

## save to csv
#np.savetxt("./data/temp_predicted_0.csv", predicted, delimiter=",")
#np.savetxt("./data/temp_y_test_0.csv", y_test, delimiter=",")
        
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
## save to csv
#np.savetxt("./data/temp_generated_0.csv", eig_generated1, delimiter=",")     
#print time.time() - t0         
