# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:27:36 2016

@author: kezhili
"""
import os

#for root, dirs, files in os.walk('Z:/DLWeights/nas207-1/experimentBackup/from pc207-18/!worm_videos/from pc207-12/misc_videos'):
#     for file in files:
#        with open(os.path.join(root, file), "r") as auto:
#            a = auto.read(1)

root = 'Z:/DLWeights/nas207-1/experimentBackup/from pc207-7/!worm_videos/copied_from_pc207-8/Andre/03-03-11/'
for path, subdirs, files in os.walk(root):
    for name in files:
        print os.path.join(path, name)            