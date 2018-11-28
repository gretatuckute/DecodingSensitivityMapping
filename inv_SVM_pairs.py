# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:18:46 2018

@author: Greta
"""

import pickle 
import os
import scipy.io as sio
import numpy as np

def getDualCoef(SVMpickle):
    '''SVMpickle is a filename of the SVM clf, e.g. 'SVM.pckl'.
    '''
    with open(SVMpickle, 'rb') as pickleFile:
        SVM1 = pickle.load(pickleFile)
    
    dc = SVM1.dual_coef_
    s_array = SVM1.support_
    
    dc = np.squeeze(dc)
    
    dc_fullsize=np.zeros(4830)
    cnt = 0
    for i in range(4830):
        if i in s_array:
            dc_fullsize[i] = dc[cnt]
            cnt += 1
        else:
            continue
        
    return dc_fullsize

def sensMap(gamma, dualcoef, eeg):
    '''
    Needed:
        Gamma used for training the SVM classifier
        Dual coefficient matrix (alpha)
        EEG (X)
        Training kernel matrix, Mexpgamma (K) 
        
    
    '''
    ######### Matrices and gamma SVM evaluation ##############
    M = distance.pdist(X,'euclidean')
    #Msquare = distance.squareform(M)
    Mexp = np.exp(-M**2)
    Mexpsquare = distance.squareform(Mexp)
    
    Mexp_gamma = np.exp(0.00025*(-(np.square(M)))) # Is this correct? 0.00025 should not be squared?
    Mexpsquare_gamma = distance.squareform(Mexp_gamma)

Xt = np.transpose(X)

# COMPUTE SENSITIVITY
# map=X*diag(alpha)*K−X*diag(alpha*K);
# s=sum(map.*map,2)/numel(alpha);

X1 = Xt
K = Mexpsquare_gamma
alpha = empty #Python can't use the alpha shape (1, 9660)
alpha1 = np.squeeze(alpha)

map1 = np.matmul(X1,np.matmul(np.diag(alpha1),K))-(np.matmul(X1,(np.diag(np.matmul(alpha1,K)))))
s = np.sum(np.square(map1),axis=1)/np.size(alpha) #Px1 vector

s_res = np.reshape(s,[32,60])
