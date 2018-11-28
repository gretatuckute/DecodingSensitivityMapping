# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 18:27:13 2018

@author: Greta
"""

import pickle 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets
import os
import pandas as pd
from scipy.spatial import distance
import scipy.io as sio

os.chdir('C:/Users/Greta/Documents/GitHub/decoding/data/ASR/')

# LOAD EEG DATA #
ASR = sio.loadmat('ASRinterp')
#X = ASR['EV_Z']
y = ASR['Animate']

# LOAD EEG WAVELET DATA #
W = sio.loadmat('wavelet_X')
W = W['X_W']

######### Matrices and gamma SVM evaluation ##############
M = distance.pdist(stdtest,'euclidean')
#Msquare = distance.squareform(M)
Mexp = np.exp(-M**2)
Mexpsquare = distance.squareform(Mexp)

Mexp_gamma = np.exp((1/192)*(-(np.square(M))))
Mexpsquare_gamma = distance.squareform(Mexp_gamma) # THIS SHOULD BE THE NxN TRAINING KERNEL MATRIX

plt.matshow(Mexpsquare_gamma)
plt.matshow(Mexpsquare)

# Gamma dynamic range is 0.05 to 10 

##### Testing on model with first subject out as test ######
# First iteration of my SVM crossvalidation with leave one subject out

ii = 1 # for checking which variables I have

cv = list(range(0,len(y),690))
test = list(range(ii, ii+690))
train = np.delete(list(range(0, len(y))), test, 0)

X=X[train]



###### Transpose X to achieve training examples in columns ######
Xt = np.transpose(X)

# COMPUTE SENSITIVITY
# map=X*diag(alpha)*K−X*diag(alpha*K);
# s=sum(map.*map,2)/numel(alpha);

X1 = Xt
K = Mexpsquare_gamma
alpha = dualcoef #Python can't use the alpha shape (1, 9660)
alpha1 = np.squeeze(alpha)

map1 = np.matmul(X1,np.matmul(np.diag(alpha1),K))-(np.matmul(X1,(np.diag(np.matmul(alpha1,K)))))
s = np.sum(np.square(map1),axis=1)/np.size(alpha) #Px1 vector

plt.plot(s)

s_res = np.reshape(s,[32,50])
s_res1 = np.reshape(s,[50,32])

# 1600 is the resampled EEG signal. 
