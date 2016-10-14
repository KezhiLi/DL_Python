# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:24:21 2016

@author: kezhili
"""

import tables
import h5py
import numpy as np
import string
import matplotlib.pyplot as plt

hdf5_file_name = r'Z:\Results\nas207-1\experimentBackup\from pc207-7\!worm_videos\copied_from_pc207-8\Andre\03-03-11\800 MY16 on food L_2011_03_03__16_52___3___11_skeletons.hdf5'
# 764 ED3049 on food L_2011_03_03__15_44___3___7


#dataset_name = ''

with tables.File(hdf5_file_name, 'r') as fid:
    skeletons = fid.get_node('/skeleton')[:]
    
with h5py.File(hdf5_file_name, 'r') as fid:
    skeletons2 = fid['/skeleton'][:]
    
ske_x = skeletons[:,:4,0]
ske_y = skeletons[:,:4,1]

ske_vec = skeletons[:,0:-1:4,:]- skeletons[:,3::4,:]