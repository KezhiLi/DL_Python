# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 13:46:13 2016

@author: kezhili
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 17:17:02 2016

@author: kezhili
"""
import open_worm_analysis_toolbox as mv
import matplotlib.pyplot as plt
import csv
import h5py
import os
import numpy as np

#mat_loc = "Z:/DLWeights/eig_catagory_Straits/N2/"
#cur_file = 'C:/Users/kezhili/Documents/Python Scripts/data/FromAWS/N2/N2_simulated_7_260_260_260_260_7_600ep.csv'
mat_loc = "Z:/DLWeights/eig_catagory_Straits/wild-isolate/"
cur_file = 'C:/Users/kezhili/Documents/Python Scripts/data/FromAWS/wild-isolate/wild-isolate_simulated_7_260_260_260_260_7_600ep.csv'

feature_names = ['width','length','loc_vel_hea_tip_spe','loc_vel_hea_spe','loc_vel_mid_spe','loc_vel_tai_spe',
    'loc_mot_eve_for_eve_dur','loc_mot_eve_for_fre','loc_mot_eve_pau_eve_dur','loc_mot_eve_pau_fre',
    'loc_mot_eve_bac_eve_dur','loc_mot_eve_bac_fre','loc_ome_tur_fre','loc_ups_tur_fre',
    'pat_ran','pat_cur','pos_ecc','pos_amp_rat','pos_tra_len','pos_coi_fre', 'pos_kin', 
    'pos_ben_hea_mea', 'pos_ben_nec_mea', 'pos_ben_mid_mea', 'pos_ben_hip_mea', 'pos_ben_tai_mea']
         
feature_field = ['morphology.width.midbody','morphology.length','locomotion.velocity.head_tip.speed','locomotion.velocity.head.speed',
                 'locomotion.velocity.midbody.speed','locomotion.velocity.tail.speed',
                 'locomotion.motion_events.forward.event_durations','locomotion.motion_events.forward.frequency',
                 'locomotion.motion_events.paused.event_durations','locomotion.motion_events.paused.frequency',
                 'locomotion.motion_events.backward.event_durations','locomotion.motion_events.backward.frequency',
                 'locomotion.omega_turns.frequency','locomotion.upsilon_turns.frequency',
                 'path.range', 'path.curvature', 'posture.eccentricity', 'posture.amplitude_ratio',
                 'posture.track_length','posture.coils.frequency','posture.kinks','posture.bends.head.mean',
                 'posture.bends.neck.mean','posture.bends.midbody.mean','posture.bends.hips.mean','posture.bends.tail.mean']
fea_no = len(feature_names) 
file_ind = 0
             
#for root, dirs, files in os.walk("C:/Users/kezhili/Documents/Python Scripts/data/FromAWS/7-200-200-200-7/1000ep/"):
for root, dirs, files in os.walk(mat_loc):
#    cur_file = 'C:/Kezhi/MyCode!!!/Simulated_Worm/tbh-1_simulated_7_260_260_260_260_7_600ep.csv'    
    
    for file in files:        
        if file.endswith(".mat"):
            print(file)
            file_ind = file_ind +1
            
            # Load a "basic" worm from a file
            #bw = mv.BasicWorm.from_schafer_file_factory("example_contour_and_skeleton_info.mat")
            bw = mv.BasicWorm.from_schafer_file_factory(root+file)
            
            # Normalize the basic worm
            nw = mv.NormalizedWorm.from_BasicWorm_factory(bw)
            ## Plot this normalized worm    
            #wp = mv.NormalizedWormPlottable(nw, interactive=False)
            #wp.show()
            # Obtain features
            wf = mv.WormFeatures(nw)
            
            wf_ess = {}
            
            # if read None data, use 0 instead
            for ind in range(fea_no):
                wf_ess[feature_field[ind]] = wf._features[feature_field[ind]].value
                if (wf_ess[feature_field[ind]]) is None:
                    wf_ess[feature_field[ind]] = 0
            
            # save the essential features to hdf5 file        
            with h5py.File(root+file[:-9]+'_esFea.h5', 'w') as hf:
                for k, v in wf_ess.items():
                    hf.create_dataset(k, data=v)
#            plt.plot(wf_ess[feature_field[21]])
            
            if file_ind == 1:
            # write the mean of features to a .csv file    
                with open(cur_file, 'wb') as csvfile:
                
                # write the column names
                    csvwriter = csv.DictWriter(csvfile, fieldnames=['file_name']+feature_field)
                    #                # jump the header(1st row) to the next row  
                    csvwriter.writeheader()
                    csvwriter.writerow({
                        # write features in each column
                        'file_name' : file[:-9],
                        feature_field[0]: np.nanmedian(wf_ess[feature_field[0]]), 
                        feature_field[1]: np.nanmedian(wf_ess[feature_field[1]]),
                        feature_field[2]: np.nanmedian(wf_ess[feature_field[2]])/3,  # because of frame rate (simulated worm is around 10frames/s, comparing to 30 frames/s)
                        feature_field[3]: np.nanmedian(wf_ess[feature_field[3]])/3,  # because of frame rate
                        feature_field[4]: np.nanmedian(wf_ess[feature_field[4]])/3,  # because of frame rate
                        feature_field[5]: np.nanmedian(wf_ess[feature_field[5]])/3,  # because of frame rate
                        feature_field[6]: np.nanmedian(wf_ess[feature_field[6]]),
                        feature_field[7]: np.nanmedian(wf_ess[feature_field[7]]),
                        feature_field[8]: np.nanmedian(wf_ess[feature_field[8]]),
                        feature_field[9]: np.nanmedian(wf_ess[feature_field[9]]),
                        feature_field[10]: np.nanmedian(wf_ess[feature_field[10]]),
                        feature_field[11]: np.nanmedian(wf_ess[feature_field[11]]),
                        feature_field[12]: np.nanmedian(wf_ess[feature_field[12]]),
                        feature_field[13]: np.nanmedian(wf_ess[feature_field[13]]),
                        feature_field[14]: np.nanmedian(wf_ess[feature_field[14]]),
                        feature_field[15]: np.nanmedian(wf_ess[feature_field[15]]),
                        feature_field[16]: np.nanmedian(wf_ess[feature_field[16]]),
                        feature_field[17]: np.nanmedian(wf_ess[feature_field[17]]),
                        feature_field[18]: np.nanmedian(wf_ess[feature_field[18]]),
                        feature_field[19]: np.nanmedian(wf_ess[feature_field[19]]),
                        feature_field[20]: np.nanmedian(wf_ess[feature_field[20]]),
                        feature_field[21]: np.nanmedian(wf_ess[feature_field[21]]),
                        feature_field[22]: np.nanmedian(wf_ess[feature_field[22]]),
                        feature_field[23]: np.nanmedian(wf_ess[feature_field[23]]),
                        feature_field[24]: np.nanmedian(wf_ess[feature_field[24]]),
                        feature_field[25]: np.nanmedian(wf_ess[feature_field[25]])
                    })
            else:
                with open(cur_file, 'ab') as csvfile:
                    csvwriter = csv.DictWriter(csvfile, fieldnames=['file_name']+feature_field)
                    csvwriter.writerow({
                        # write features in each column
                        'file_name' : file[:-9],
                        feature_field[0]: np.nanmedian(wf_ess[feature_field[0]]), 
                        feature_field[1]: np.nanmedian(wf_ess[feature_field[1]]),
                        feature_field[2]: np.nanmedian(wf_ess[feature_field[2]])/3,  # because of frame rate
                        feature_field[3]: np.nanmedian(wf_ess[feature_field[3]])/3,  # because of frame rate
                        feature_field[4]: np.nanmedian(wf_ess[feature_field[4]])/3,  # because of frame rate
                        feature_field[5]: np.nanmedian(wf_ess[feature_field[5]])/3,  # because of frame rate
                        feature_field[6]: np.nanmedian(wf_ess[feature_field[6]]),
                        feature_field[7]: np.nanmedian(wf_ess[feature_field[7]]),
                        feature_field[8]: np.nanmedian(wf_ess[feature_field[8]]),
                        feature_field[9]: np.nanmedian(wf_ess[feature_field[9]]),
                        feature_field[10]: np.nanmedian(wf_ess[feature_field[10]]),
                        feature_field[11]: np.nanmedian(wf_ess[feature_field[11]]),
                        feature_field[12]: np.nanmedian(wf_ess[feature_field[12]]),
                        feature_field[13]: np.nanmedian(wf_ess[feature_field[13]]),
                        feature_field[14]: np.nanmedian(wf_ess[feature_field[14]]),
                        feature_field[15]: np.nanmedian(wf_ess[feature_field[15]]),
                        feature_field[16]: np.nanmedian(wf_ess[feature_field[16]]),
                        feature_field[17]: np.nanmedian(wf_ess[feature_field[17]]),
                        feature_field[18]: np.nanmedian(wf_ess[feature_field[18]]),
                        feature_field[19]: np.nanmedian(wf_ess[feature_field[19]]),
                        feature_field[20]: np.nanmedian(wf_ess[feature_field[20]]),
                        feature_field[21]: np.nanmedian(wf_ess[feature_field[21]]),
                        feature_field[22]: np.nanmedian(wf_ess[feature_field[22]]),
                        feature_field[23]: np.nanmedian(wf_ess[feature_field[23]]),
                        feature_field[24]: np.nanmedian(wf_ess[feature_field[24]]),
                        feature_field[25]: np.nanmedian(wf_ess[feature_field[25]])
                    })
                    
    
    
