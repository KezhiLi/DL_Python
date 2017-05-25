# -*- coding: utf-8 -*-
"""
Created on Wed Apr 05 17:23:10 2017

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
from tabulate import tabulate 
import csv

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
    
    
file_count = 0 
file_no = 0
table_ori = [["test_name","test_path",0,0,0,0,0,0,0]]
table = [["test_name","test_path",0,0,0,0,0,0,0]]
table_full_ori = [["test_name","test_path",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
table_full = [["test_name","test_path",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
root = 'X:/Kezhi/Classifier_DL/testing_nips/'
root_simulate = 'Z:/Ken_Samples/simulated/'
count_strain = [0,0,0,0,0,0,0]
for path, dirs, files in os.walk(root):
    for name in files:
        print file_count
        file_count = file_count + 1
        try:
            if name.endswith(("_eig.hdf5")) and ('on food' in name) and (('unc-9' in name) or \
            ('tbh-1' in name) or ('N2' in name)or \
            ('CB4852' in name)or ('CB4856' in name)or \
            ('ED3017' in name)or ('LSJ1' in name)): 
                print os.path.join(path, name) 
                with h5py.File(os.path.join(path, name), 'r') as fid:
                    eig_coef = fid['/eig_coef'][:]  
               
                columns = ['a','b','c','d','e','f','g']   
                data = pd.DataFrame(eig_coef, columns =  columns)  
    
                n_prev = 50    
                len_data = len(data)
                start_fraction = 0.9
                     
                (X_train, y_train), (X_test, y_test) = train_test_split(data,start_fraction,n_prev)  # retrieve data
                del data
                
                table[file_no][0] = name
                table[file_no][1] = path
                table_full[file_no][0] = name
                table_full[file_no][1] = path
                
                for ii in range(7):
                    # whatever
                    X_test_LR, y_test_LR = X_test, y_test
                    if ii == 0:
                        stain =  'unc-9'  
                        if  'L_' in name:
                            X_test_LR, y_test_LR = -X_test, -y_test
                    elif ii == 1:
                        stain =  'tbh-1'  
                        if  'R_' in name:
                            X_test_LR, y_test_LR = -X_test, -y_test
                    elif ii == 2:
                        stain =  'N2'   
                        if  'R_' in name:
                            X_test_LR, y_test_LR = -X_test, -y_test     
                    elif ii == 3:
                        stain =  'CB4852'
                        if  'L_' in name:
                            X_test_LR, y_test_LR = -X_test, -y_test                            
                    elif ii == 4:
                        stain =  'CB4856'  
                        if  'L_' in name:
                            X_test_LR, y_test_LR = -X_test, -y_test
                    elif ii == 5:
                        stain =  'ED3017'   
                        if  'L_' in name:
                            X_test_LR, y_test_LR = -X_test, -y_test
                    elif ii == 6:
                        stain =  'LSJ1' 
                        if  'L_' in name:
                            X_test_LR, y_test_LR = -X_test, -y_test
                    else:
                        continue                    
                    print ii
        
                    model_name = 'C:/Users/kezhili/Documents/Python Scripts/data/FromAWS/'+stain+'/multiFile_'+stain+'_7-260-260-260-260-7_600ep.h5' 
                    model.load_weights(model_name) 
                                
                    predicted = model.predict(X_test_LR)  
                    rmse = np.sqrt(((predicted - y_test_LR) ** 2).mean(axis=0))
                    
                    table[file_no][ii+2] = rmse.mean()
                    table_full[file_no][(2+ii*7):(2+(ii+1)*7)] = rmse
                
                if 'unc-9' in name:
                    count_strain[0] = count_strain[0]+1
                elif 'tbh-1'  in name:
                    count_strain[1] = count_strain[1]+1   
                elif 'N2' in name:
                    count_strain[2] = count_strain[2]+1   
                elif 'CB4852' in name:
                    count_strain[3] = count_strain[3]+1    
                elif 'CB4856' in name:
                    count_strain[4] = count_strain[4]+1   
                elif 'ED3017' in name:
                    count_strain[5] = count_strain[5]+1   
                elif 'LSJ1' in name:
                    count_strain[6] = count_strain[6]+1     
                else:
                    continue
                    
                file_no = file_no + 1
                table = np.vstack([table, table_ori])
                table_full = np.vstack([table_full, table_full_ori])
        except:
            pass   

print tabulate(table, headers=['file_name', 'file_path','unc-9','tbh-1','N2', 'CB4852','CB4856','ED3017','LSJ1'])

with open('table_20170509.csv', 'wb') as f: 
    writer = csv.writer(f)
    writer.writerows(table)
with open('table_full_20170509.csv', 'wb') as ff: 
    writer = csv.writer(ff)
    writer.writerows(table_full)    
    
#XX = table[:,2:]
#YY = np.zeros((len(XX), len(XX[0])))
#Y_name = table[:,0]
#for ii in range(Y_name.size):
#    if 'N2' in name:
#        YY[ii,0] = 1 
#    elif 'unc-8'  in name:
#        YY[ii,1] = 1    
#    elif 'ser-6' in name:
#        YY[ii,2] = 1    
#    elif 'tdc-1' in name:
#        YY[ii,3] = 1    
#    elif 'tbh-1' in name:
#        YY[ii,4] = 1    
#    elif 'CB4856' in name:
#        YY[ii,5] = 1   
#    elif 'unc-9' in name:
#        YY[ii,6] = 1 
#    elif 'trp-4' in name:
#        YY[ii,7] = 1      
#    elif 'CB4852' in name:
#        YY[ii,8] = 1   
#    elif 'ED3017' in name:
#        YY[ii,9] = 1   
#    elif 'ED3049' in name:
#        YY[ii,10] = 1  
#    elif 'ED3054' in name:
#        YY[ii,11] = 1 
#    elif 'JU298' in name:
#        YY[ii,12] = 1 
#    elif 'JU345' in name:
#        YY[ii,13] = 1 
#    elif 'JU393' in name:
#        YY[ii,14] = 1 
#    elif 'JU438' in name:
#        YY[ii,15] = 1 
#    elif 'JU440' in name:
#        YY[ii,16] = 1 
#    elif 'LSJ1' in name:
#        YY[ii,17] = 1 
#    elif 'MY16' in name:
#        YY[ii,18] = 1 
#    else:
#        continue
#
#
## create model
#model_classifer = Sequential()
#model_classifer.add(Dense(19, 19, activation='relu'))
#model_classifer.add(Dense(19, 19, activation='relu'))
#model_classifer.add(Dense(19, 19, activation='sigmoid'))    
#
## Compile model
#model_classifer.compile(loss='binary_crossentropy', optimizer='adam')
#
## Fit the model
#model_classifer.fit(XX, YY, batch_size=10, nb_epoch=200)
#
## evaluate the model
#scores_dl = model_classifer.evaluate(XX, YY)
#print("\n%s: %.2f%%" % (model_classifer.metrics_names[1], scores_dl[1]*100))
