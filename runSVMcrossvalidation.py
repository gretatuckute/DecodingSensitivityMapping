# -*- coding: utf-8 -*-
"""
Script for running SVM classifier cross-validation in a leave-one-subject-out approach.

Runs from the command line.

#  EXAMPLE FUNCTION CALL
#	python runSVM.py -s 0

@author: Greta Tuckute, grtu@dtu.dk, November 2018

"""

#Imports
import argparse
import numpy as np
from sklearn.svm import SVC
import scipy.io as sio
import pandas as pd
import datetime

date = str(datetime.datetime.now())
date_str = date.replace(' ','-')
date_str = date_str.replace(':','.')
date_str = date_str[:-10]

# Constructing the parser and parse the arguments
parser = argparse.ArgumentParser(description='Takes subject number for test set (-s)')
parser.add_argument('-s','--subject', required=True, default=None,help='Specify which subject to leave out as test set. 0 = subject 1')
args = vars(parser.parse_args()) 

subj = args['subject']
subj = int(subj)
no_trials = 690 # Number of trials for each subject

# Load data
ASR = sio.loadmat('ASRinterp')
X = ASR['EV_Z']

y = ASR['Animate']
y = np.squeeze(y)

# Change y to 1 and -1
y = y.astype(np.int16)
np.putmask(y, y<=0, -1)
y = y.astype(np.int16)

print('============ Data Loaded ============')
print('X train shape: ' + str(X_train.shape))
print('y train shape: ' + str(y_train.shape))
print('Subject test: ' + str(subj+1))

# Parameters to iterate through
C_2d_range = [0.05, 0.25, 0.5, 1, 1.5, 1.75, 2, 2.5, 5, 10]
gamma_2d_range = [1/4000000, 1/2000000, 1/400000, 1/200000, 1/40000, 1/20000, 1/4000, 1/2000, 1/400, 1/200]

random_state = np.random.RandomState(0)

df = pd.DataFrame(columns=['Subject no.', 'C value', 'Gamma value', 'scores_train', 'scores_test'])
cv = list(range(0,len(y),no_trials)) 
ii = cv[subj]

test = list(range(ii, ii+no_trials))
train = np.delete(list(range(0, len(y))), test, 0)

X_train=X[train]
y_train=y[train]

X_test=X[test]
y_test=y[test]

count = 0

for gamma in gamma_2d_range:
    for C in C_2d_range:
        classifier = SVC(C=C, gamma=gamma, random_state=random_state)
        clf = classifier.fit(X_train, y_train)
    
        scores_train = clf.score(X_train, y_train)
        scores_test = clf.score(X_test, y_test)
        df.loc[count]=[subj+1, C, gamma, scores_train, scores_test]
    
        count += 1
        print(str(count))
        print(str(gamma))
            
df.to_excel('CV_' + str(subj+1) + '.xlsx')

    

