### Imports ###

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
import os
from scipy.spatial import distance
import scipy.io as sio

def computeSensitivityMap(X, y, C_val, gamma_val, no_channels, no_timepoints):
    """
    Function for computing a sensitivity map for an EEG-based Radial Basis Function (RBF) Support Vector Machine (SVM) classifier.       
        
    ### Inputs ###
    # X: EEG data 2d matrix containing trials as rows, and features (channels * time points) as columns.
    # y: List/NumPy array containing binary class labels, y = {-1, 1}.
    # C: SVM classifier regularization parameter. 
    # Gamma: Free parameter of the RBF kernel, SVM classifier.
    
    
    ### Outputs ###
    # s_matrix: sensitivity map matrix.
    # plt: Visualization of the sensitivity map.
    
    ### Example function call ###
    computeSensitivityMap(X, y, C_val = 1, gamma_val = 0.0005, no_channels = 32, no_timepoints = 60)

    """
    
    ### Compute SVM classifier ###
    y = np.squeeze(y)
    classifier = SVC(C=C_val, gamma=gamma_val)
    clf = classifier.fit(X, y)
    
    
    ### Extract classifier model coefficients and add zero indices ### 
    coefficients = clf.dual_coef_
    support_array = clf.support_
    
    coefficients = np.squeeze(coefficients)
    
    trials = len(X[:,0])
    features = len(X[0])
    alpha = np.zeros(trials)
    alpha[support_array] = coefficients
    alpha = np.squeeze(alpha)
    
    no_zero_indices = trials - len(support_array)
    
    ### Compute training kernal matrix, K ###
    M = distance.pdist(X,'euclidean')
    
    M_exp = np.exp(gamma_val*(-(np.square(M))))
    K = distance.squareform(M_exp) 
    
    ### Compute sensitivity map ###
    
    X = np.transpose(X) # Obtain training examples in columns for further computation

    mapping = np.matmul(X,np.matmul(np.diag(alpha),K))-(np.matmul(X,(np.diag(np.matmul(alpha,K)))))
    s = np.sum(np.square(mapping),axis=1)/np.size(alpha) 

    s_matrix = np.reshape(s,[no_channels,no_timepoints])
    # np.save('sensitivity_map.npy',s_matrix)
    
    ### Generation of sensitivity map plot ###
    
    # Examples of x- and y-axis labels
    
#    channel_vector = ['P7','P4','Cz','Pz','P3','P8','O1','O2','T8','F8','C4','F4',
#                  'Fp2','Fz','C3','F3','Fp1','T7','F7','Oz','PO3','AF3','FC5',
#                  'FC1','CP5','CP1','CP2','CP6','AF4','FC2','FC6','PO4']
#
#    time_vector = ['-100','0','100','200','300','400','500']

    plt.matshow(s_matrix)
    plt.xlabel('Time points')
    #plt.xticks(np.arange(0,no_timepoints,10),time_vector)
    #plt.yticks(np.arange(no_channels),channel_vector)
    plt.ylabel('EEG channels')
    plt.colorbar()
    plt.title('Sensitivity map SVM RBF kernel')
    # plt.show()
    
    return s_matrix, plt
    
    print('Sensitivity map computed. Number of support vectors for the classifier: {0}.'.format(len(support_array)))
