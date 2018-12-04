# -*- coding: utf-8 -*-
"""
Implementation of the NPAIRS (Strother et al., 2011) cross-validation framework for estimation of sensitivity map visualization uncertainty.

Calls the NPAIRS_functions.py script.

@author: Greta Tuckute, grtu@dtu.dk, November 2018
"""

import numpy as np
from sklearn.svm import SVC
import scipy.io as sio
import pandas as pd
import os
from NPAIRS_functions import * 

no_runs = 100

# Load data
ASR = sio.loadmat('X')
X = ASR['EV_Z']

y = ASR['Animate']
y = np.squeeze(y)

y = y.astype(np.int16)
np.putmask(y, y<=0, -1)
y = y.astype(np.int16)

cnt = 1
df = pd.DataFrame(columns=['Split 1 subjects', 'Split 2 subjects']) # Saves which subjects that were randomly drawn to compute NPAIRS

smaps = []

###### LOOP FOR NPAIRS ITERATIONS ######
for ii in range(no_runs):
    ''' Run multiple SVM pair sessions and output the difference between sensitivity maps'''
    sq_smap, split1, split2 = runPairSVM(X, y, C_val=1.5, gamma_val=0.00005)
    
    df.loc[cnt]=[split1, split2]
    cnt += 1
    
    smaps.append(sq_smap)

print('No. of sensitivity maps used for computing the mean value: ' + str(len(smaps)))

###### MEAN OVER SENSITIVITY MAPS #######
mean_smaps = np.mean(smaps,axis=0)
std_smaps = np.sqrt(mean_smaps) # Standard deviation

# Divide the std_smaps with the original sensitivity map for all 15 subjs
mean_15 = np.load('s_map_mean_NEW.npy') # The original sensitivity map has to be loaded.

effect_smap = np.divide(mean_15,std_smaps)

###### SAVE AND LOG ######
np.save('mean_smap_new_' + str(no_runs) + '.npy',mean_smaps)
np.save('std_smap_new_' + str(no_runs) + '.npy',std_smaps)
np.save('effect_smap_new_' + str(no_runs) + '.npy',effect_smap)
# Log which subjects were used in the splits 
df.to_excel('smap_split_new_' +  str(no_runs) + '.xlsx')

