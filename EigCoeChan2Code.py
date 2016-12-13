# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 16:40:14 2016

@author: kezhili
"""

from sklearn.cluster import KMeans
from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dropout  
from keras.layers.recurrent import LSTM
import pandas as pd  
import time
import h5py
import numpy as np
from random import random
import matplotlib.pyplot as plt
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
    
t0 = time.time() 

columns = ['a', 'b','c','d','e','f','g']   
eig_diff_all = []

file_no = 0
root = 'Z:/DLWeights/nas207-1/experimentBackup/from pc207-7/!worm_videos/copied_from_pc207-8/Andre/'
#root = 'Z:\DLWeights\nas207-1\experimentBackup\from pc207-7\!worm_videos\copied_from_pc207-8\Andre\'
for path, subdirs, files in os.walk(root):
    for name in files:
        if name.endswith(".hdf5")&(file_no<50):  
            cur_file = os.path.join(path, name).replace("\\","/") 
            print cur_file
            print file_no
            file_no = file_no + 1
            
            with h5py.File(cur_file, 'r') as fid:
                eig_coef = fid['/eig_coef'][:]
       #         eig_coef = round(eig_coef,3)
                eig_coef_dif = eig_coef[1:,:] - eig_coef[:-1,:]
                if file_no ==1:
                    eig_diff_all = eig_coef_dif
                else:
                    eig_diff_all = np.concatenate((eig_diff_all,eig_coef_dif),axis=0)
        
# ply.plot(eig_coef_dif[:,0])

        
data_diff_all = pd.DataFrame(eig_diff_all, columns =  columns)  

del eig_diff_all

kmeans_diff = KMeans(n_clusters=1000, random_state=0).fit(data_diff_all)
#kmeans.labels_
#array([0, 0, 0, 1, 1, 1], dtype=int32)
#>>> kmeans.predict([[0, 0], [4, 4]])
#array([0, 1], dtype=int32)
#>>> kmeans.cluster_centers_
#array([[ 1.,  2.],
#       [ 4.,  2.]])

from sklearn.externals import joblib
joblib.dump(kmeans_diff, "kmeans_AndreTo1000.model")
#kmeans_diff = joblib.load("kmeans_model.m")





