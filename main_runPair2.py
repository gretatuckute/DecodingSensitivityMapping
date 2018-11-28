# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 20:31:15 2018

@author: Greta
"""

import numpy as np
from sklearn.svm import SVC
import scipy.io as sio
import pickle 
import pandas as pd
import datetime
import os
from functionsPair2 import * 

no_runs = 100

# Load data
#os.chdir('C:/Users/Greta/Documents/GitHub/decoding/data/ASR/')

os.chdir('/home/grtu/decoding/')

ASR = sio.loadmat('ASRinterp')
X = ASR['EV_Z']

y = ASR['Animate']
y = np.squeeze(y)
y = y.astype(np.int16)
np.putmask(y, y<=0, -1)
y = y.astype(np.int16)

cnt = 1
df = pd.DataFrame(columns=['Split 1 subjects', 'Split 2 subjects'])

smaps = []
for ii in range(no_runs):
    ''' Run multiple SVM pair sessions and output the difference between sensitivity maps'''
    sq_smap, split1, split2 = runPairSVM(X, y, C_val=1.5, gamma_val=0.00005)
    
    df.loc[cnt]=[split1, split2]
    cnt += 1
    
    smaps.append(sq_smap)

print('Length of smaps (no. of smaps that are used for the mean value: ' + str(len(smaps)))

###### MEAN OVER SENSITIVITY MAPS #######
mean_smaps = np.mean(smaps,axis=0)
std_smaps = np.sqrt(mean_smaps) # This is the std 

# Divide the std_smaps with the original sensitivity map for all 15 subjs
mean_15 = np.load('s_map_mean_NEW.npy')

effect_smap = np.divide(mean_15,std_smaps)

###### SAVE AND LOG ######
os.chdir('/home/grtu/decoding/SVM_pairs/')

np.save('mean_smap_new_' + str(no_runs) + '.npy',mean_smaps)
np.save('std_smap_new_' + str(no_runs) + '.npy',std_smaps)
np.save('effect_smap_new_' + str(no_runs) + '.npy',effect_smap)
# Log which subjects were used in the splits 
df.to_excel('smap_split_new_' +  str(no_runs) + '.xlsx')

