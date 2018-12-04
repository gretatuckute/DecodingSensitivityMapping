# -*- coding: utf-8 -*-
"""
Implementation of the NPAIRS (Strother et al., 2011) cross-validation framework for estimation of sensitivity map visualization uncertainty.

Called from the NPAIRS_main.py script.

@author: Greta Tuckute, grtu@dtu.dk, November 2018

"""

import numpy as np
import pickle
import os
import pandas as pd
from sklearn.svm import SVC
import datetime
from scipy.spatial import distance


def makeSplit(X, y):
    ''' Creates two random partitions of subjects (7*2) from the EEG (X) and labels (y) from a total of 15 subjects.
    
    Inputs:
        X: EEG data 2d matrix containing trials as rows, and features (channels * time points) as columns.
        y: List/NumPy array containing binary class labels, y = {-1, 1}.
    
    Outputs:
        X1, X2, y1 and y2: EEG and labels for two different partitions.
        lst1 and lst2: Subjects that were randomly drawn for the two partitions.
    
    '''
    
    no_trials = 690 # Number of trials for each subject
    
    cv = list(range(0,len(y),no_trials))
    subj_dict = dict(zip(cv,list(range(1,16)))) # Number of subjects. 15 subjects in this example.
    fourteen = np.random.choice(cv, 14, replace = False)
    split1 = fourteen[0:7]
    split2 = fourteen[7:14]

    subs1 = []
    labels1 = []
    subs2 = []
    labels2 = []
    
    for count, ii in enumerate(split1):
        slice_idx = list(range(ii, ii+no_trials))
        eeg = X[slice_idx]
        labels = y[slice_idx]
        subs1.append(eeg)
        labels1.append(labels)
        
        if count == 6:
            subs1 = np.concatenate(subs1,axis=0)
            labels1 = np.concatenate(labels1,axis=0)
            
    for count, ii in enumerate(split2):
        slice_idx = list(range(ii, ii+no_trials))
        eeg = X[slice_idx]
        labels = y[slice_idx]
        subs2.append(eeg)
        labels2.append(labels)
        
        if count == 6:
            subs2 = np.concatenate(subs2,axis=0)
            labels2 = np.concatenate(labels2,axis=0)
            
    # Log subject splits
    lst1 = []
    for k in split1:
        s1 = subj_dict.get(k)
        lst1.append(s1)
        
    lst2 = []
    for k in split2:
        s2 = subj_dict.get(k)
        lst2.append(s2)
            
    return subs1, labels1, subs2, labels2, lst1, lst2

def runPairSVM(X, y, C_val=1.5, gamma_val=0.00005):
    ''' 
    Inputs:
        X: EEG data 2d matrix containing trials as rows, and features (channels * time points) as columns.
        y: List/NumPy array containing binary class labels, y = {-1, 1}.
        C: SVM classifier regularization parameter. 
        Gamma: Free parameter of the RBF kernel, SVM classifier.
    
    Returns
        sq_smap: The difference between the two sensitivity maps, squared.
        split1, split2: Subjects used for the two splits.
    '''
    
    no_channels = 32
    no_timepoints = 60
    
    X1, y1, X2, y2, split1, split2 = makeSplit(X,y)
    
    no_trials = len(X1[:,0])
    
    print('============ Data Loaded ============')
    print('Split 1, X1 shape: ' + str(X1.shape))
    print('Split 1, y1 shape: ' + str(y1.shape))
    print('Split 2, X2 shape: ' + str(X2.shape))
    print('Split 2, y2 shape: ' + str(y2.shape))
    print('Subjects split 1: ' + str(split1))
    print('Subjects split 2: ' + str(split2))
    
    random_state = np.random.RandomState(0)
    
    # Train SVM on 7 subjs, and 7 subjs again
    classifier1 = SVC(random_state=random_state, C=C_val, gamma=gamma_val)
    clf1 = classifier1.fit(X1, y1)
    
    classifier2 = SVC(random_state=random_state, C=C_val, gamma=gamma_val)
    clf2 = classifier2.fit(X2, y2)
    
    print('Ran two SVM classifiers, first one for subjects: ' + str(split1) + 'and second one for subjects: ' + str(split2))
    
    ####### MAKE SENSITIVITY MAP - clf1 ########
    M1 = distance.pdist(X1,'euclidean')

    Mexp_gamma1 = np.exp(gamma_val*(-(np.square(M1))))
    Mexpsquare_gamma1 = distance.squareform(Mexp_gamma1)
    
    Xt1 = np.transpose(X1)
    K1 = Mexpsquare_gamma1
    
    dc1 = clf1.dual_coef_
    s_array1 = clf1.support_
    
    dc1 = np.squeeze(dc1)
    
    seq1 = np.zeros(no_trials) 
    seq1[s_array1] = dc1
    
    alpha1 = np.squeeze(seq1)
    
    map1 = np.matmul(Xt1,np.matmul(np.diag(alpha1),K1))-(np.matmul(Xt1,(np.diag(np.matmul(alpha1,K1)))))
    s1 = np.sum(np.square(map1),axis=1)/np.size(alpha1) #Px1 vector
    
    s_res1 = np.reshape(s1,[no_channels,no_timepoints])
    
    ####### MAKE SENSITIVITY MAP - clf2 ########
    M2 = distance.pdist(X2,'euclidean')

    Mexp_gamma2 = np.exp(gamma_val*(-(np.square(M2))))
    Mexpsquare_gamma2 = distance.squareform(Mexp_gamma2)
    
    Xt2 = np.transpose(X2)
    K2 = Mexpsquare_gamma2
    
    dc2 = clf2.dual_coef_
    s_array2 = clf2.support_
    
    dc2 = np.squeeze(dc2)
    
    seq2 = np.zeros(no_trials)
    seq2[s_array2] = dc2
    
    alpha2 = np.squeeze(seq2)
    
    map2 = np.matmul(Xt2,np.matmul(np.diag(alpha2),K2))-(np.matmul(Xt2,(np.diag(np.matmul(alpha2,K2)))))
    s2 = np.sum(np.square(map2),axis=1)/np.size(alpha2) #Px1 vector
    
    s_res2 = np.reshape(s2,[no_channels,no_timepoints])
    
    print('Two sensitivity maps created!')

    ####### FIND DIFFERENCE BETWEEN SENS MAPS AND SQUARE #######
    
    diff_smap = np.subtract(s_res1,s_res2) 
    
    sq_smap = np.square(diff_smap)
    
    return sq_smap, split1, split2


    
    
    
    
    