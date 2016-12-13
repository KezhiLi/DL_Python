# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 15:29:59 2016

@author: kezhili
"""
import pandas as pd
import h5py
import csv
import os
import numpy as np

feature_field = ['morphology.width.midbody','morphology.length','locomotion.velocity.head_tip.speed','locomotion.velocity.head.speed',
                 'locomotion.velocity.midbody.speed','locomotion.velocity.tail.speed',
                 'locomotion.motion_events.forward.event_durations','locomotion.motion_events.forward.frequency',
                 'locomotion.motion_events.paused.event_durations','locomotion.motion_events.paused.frequency',
                 'locomotion.motion_events.backward.event_durations','locomotion.motion_events.backward.frequency',
                 'locomotion.omega_turns.frequency','locomotion.upsilon_turns.frequency',
                 'path.range', 'path.curvature', 'posture.eccentricity', 'posture.amplitude_ratio',
                 'posture.track_length','posture.coils.frequency','posture.kinks','posture.bends.head.mean',
                 'posture.bends.neck.mean','posture.bends.midbody.mean','posture.bends.hips.mean','posture.bends.tail.mean']


csv_file = r'C:\Users\kezhili\Documents\GitHub\Multiworm_Tracking\MWTracker\auxFiles\features_names.csv'

df = pd.read_csv(csv_file)
df.index = df['feat_name_obj']
#df = df.loc[feature_field]



feat_file = r'C:\Users\kezhili\Documents\Python Scripts\data\FromAWS\7-200-200-200-200-7\07-03-11\600epochs\no_noise\247 JU438 on food L_2011_03_07__12_53___3___7_features.hdf5'

    
feature_names = ['width','length','loc_vel_hea_tip_spe','loc_vel_hea_spe','loc_vel_mid_spe','loc_vel_tai_spe',
    'loc_mot_eve_for_eve_dur','loc_mot_eve_for_fre','loc_mot_eve_pau_eve_dur','loc_mot_eve_pau_fre',
    'loc_mot_eve_bac_eve_dur','loc_mot_eve_bac_fre','loc_ome_tur_fre','loc_ups_tur_fre',
    'pat_ran','pat_cur','pos_ecc','pos_amp_rat','pos_tra_len','pos_coi_fre', 'pos_kin', 
    'pos_ben_hea_mea', 'pos_ben_nec_mea', 'pos_ben_mid_mea', 'pos_ben_hip_mea', 'pos_ben_tai_mea']
    
fea_no = len(feature_names) 
file_ind = 0

             
for root, dirs, files in os.walk("C:/Users/kezhili/Documents/Python Scripts/data/FromAWS/7-200-200-200-200-7/07-03-11/600epochs/no_noise/"):
    for feat_file in files:        
        if feat_file.endswith("_features.hdf5"):
            print(feat_file)
            file_ind = file_ind +1
            
            with pd.HDFStore(root+feat_file, 'r') as fid:
                features_timeseries = fid['/features_timeseries']
            
            all_feats = {}
            for feat in feature_field:
                df_row = df.loc[feat]
                
                if df_row['is_time_series'] == 1:
                    dat = features_timeseries[df_row['feat_name_table']]
                    if dat is None:
                        dat = 0
                else:
                    with h5py.File(root+feat_file, 'r') as fid:
                        dat = fid['/'.join(['features_events', 'worm_1', df_row['feat_name_table']])][:]
                all_feats[feat] = dat
            
            wf_ess = {}

            if file_ind == 1:
            # write the mean of features to a .csv file    
                with open('C:/Kezhi/MyCode!!!/Simulated_Worm/features_mean_real1.csv', 'wb') as csvfile:
                
                # write the column names
                    csvwriter = csv.DictWriter(csvfile, fieldnames=['file_name']+feature_field)
                    #                # jump the header(1st row) to the next row  
                    csvwriter.writeheader()
                    csvwriter.writerow({
                        # write features in each column
                        'file_name' : feat_file[:-14],
                        feature_field[0]: np.nanmean(all_feats[feature_field[0]]), 
                        feature_field[1]: np.nanmean(all_feats[feature_field[1]]),
                        feature_field[2]: np.nanmean(all_feats[feature_field[2]]),
                        feature_field[3]: np.nanmean(all_feats[feature_field[3]]),
                        feature_field[4]: np.nanmean(all_feats[feature_field[4]]),
                        feature_field[5]: np.nanmean(all_feats[feature_field[5]]),
                        feature_field[6]: np.nanmean(all_feats[feature_field[6]]),
                        feature_field[7]: np.nanmean(all_feats[feature_field[7]]),
                        feature_field[8]: np.nanmean(all_feats[feature_field[8]]),
                        feature_field[9]: np.nanmean(all_feats[feature_field[9]]),
                        feature_field[10]: np.nanmean(all_feats[feature_field[10]]),
                        feature_field[11]: np.nanmean(all_feats[feature_field[11]]),
                        feature_field[12]: np.nanmean(all_feats[feature_field[12]]),
                        feature_field[13]: np.nanmean(all_feats[feature_field[13]]),
                        feature_field[14]: np.nanmean(all_feats[feature_field[14]]),
                        feature_field[15]: np.nanmean(all_feats[feature_field[15]]),
                        feature_field[16]: np.nanmean(all_feats[feature_field[16]]),
                        feature_field[17]: np.nanmean(all_feats[feature_field[17]]),
                        feature_field[18]: np.nanmean(all_feats[feature_field[18]]),
                        feature_field[19]: np.nanmean(all_feats[feature_field[19]]),
                        feature_field[20]: np.nanmean(all_feats[feature_field[20]]),
                        feature_field[21]: np.nanmean(all_feats[feature_field[21]]),
                        feature_field[22]: np.nanmean(all_feats[feature_field[22]]),
                        feature_field[23]: np.nanmean(all_feats[feature_field[23]]),
                        feature_field[24]: np.nanmean(all_feats[feature_field[24]]),
                        feature_field[25]: np.nanmean(all_feats[feature_field[25]])
                    })
            else:
                with open('C:/Kezhi/MyCode!!!/Simulated_Worm/features_mean_real1.csv', 'ab') as csvfile:
                    csvwriter = csv.DictWriter(csvfile, fieldnames=['file_name']+feature_field)
                    csvwriter.writerow({
                        # write features in each column
                        'file_name' : feat_file[:-14],
                        feature_field[0]: np.nanmean(all_feats[feature_field[0]]), 
                        feature_field[1]: np.nanmean(all_feats[feature_field[1]]),
                        feature_field[2]: np.nanmean(all_feats[feature_field[2]]),
                        feature_field[3]: np.nanmean(all_feats[feature_field[3]]),
                        feature_field[4]: np.nanmean(all_feats[feature_field[4]]),
                        feature_field[5]: np.nanmean(all_feats[feature_field[5]]),
                        feature_field[6]: np.nanmean(all_feats[feature_field[6]]),
                        feature_field[7]: np.nanmean(all_feats[feature_field[7]]),
                        feature_field[8]: np.nanmean(all_feats[feature_field[8]]),
                        feature_field[9]: np.nanmean(all_feats[feature_field[9]]),
                        feature_field[10]: np.nanmean(all_feats[feature_field[10]]),
                        feature_field[11]: np.nanmean(all_feats[feature_field[11]]),
                        feature_field[12]: np.nanmean(all_feats[feature_field[12]]),
                        feature_field[13]: np.nanmean(all_feats[feature_field[13]]),
                        feature_field[14]: np.nanmean(all_feats[feature_field[14]]),
                        feature_field[15]: np.nanmean(all_feats[feature_field[15]]),
                        feature_field[16]: np.nanmean(all_feats[feature_field[16]]),
                        feature_field[17]: np.nanmean(all_feats[feature_field[17]]),
                        feature_field[18]: np.nanmean(all_feats[feature_field[18]]),
                        feature_field[19]: np.nanmean(all_feats[feature_field[19]]),
                        feature_field[20]: np.nanmean(all_feats[feature_field[20]]),
                        feature_field[21]: np.nanmean(all_feats[feature_field[21]]),
                        feature_field[22]: np.nanmean(all_feats[feature_field[22]]),
                        feature_field[23]: np.nanmean(all_feats[feature_field[23]]),
                        feature_field[24]: np.nanmean(all_feats[feature_field[24]]),
                        feature_field[25]: np.nanmean(all_feats[feature_field[25]])
                    })
                    