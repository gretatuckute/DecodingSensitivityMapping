# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:23:18 2018

@author: Greta, grtu@dtu.dk

#EXAMPLE FUNCTION CALL
#	python runSVM.py -s 0

"""

#Imports
import argparse
import numpy as np
from sklearn.svm import SVC
import scipy.io as sio
import pickle 
import pandas as pd
import datetime

date = str(datetime.datetime.now())
date_str = date.replace(' ','-')
date_str = date_str.replace(':','.')
date_str = date_str[:-10]

#Constructing the parser and parse the arguments
parser = argparse.ArgumentParser(description='Takes subject number for test set (-s)')
parser.add_argument('-s','--subject', required=True, default=None,help='Specify which subject to leave out as test set. 0 = subject 1')
args = vars(parser.parse_args()) 

subj = args['subject']
subj = int(subj)

# Load data
#os.chdir('C:/Users/Greta/Documents/GitHub/decoding/data/ASR/')
ASR = sio.loadmat('ASRinterp')
X = ASR['EV_Z']

y = ASR['Animate']
y = np.squeeze(y)

# Change y to 1 and -1
#y[y < 1] = -1
y = y.astype(np.int16)
np.putmask(y, y<=0, -1)
y = y.astype(np.int16)


# Parameters to iterate through
C_2d_range = [0.05, 0.25, 0.5, 1, 1.5, 1.75, 2, 2.5, 5, 10]
#C_2d_range = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 1.5, 2.5, 5, 10]
# C_2d_range = [0.1, 0.5, 1, 1.5, 5]
#gamma_2d_range = [0.00005, 0.00025, 0.0005, 0.00075, 0.001]
gamma_2d_range = [1/4000000, 1/2000000, 1/400000, 1/200000, 1/40000, 1/20000, 1/4000, 1/2000, 1/400, 1/200]

random_state = np.random.RandomState(0)

#df = pd.DataFrame(columns=['Subject no.', 'C value', 'scores_train', 'scores_test'])
df = pd.DataFrame(columns=['Subject no.', 'C value', 'Gamma value', 'scores_train', 'scores_test'])
cv = list(range(0,len(y),690))
ii = cv[subj]

test = list(range(ii, ii+690))
train = np.delete(list(range(0, len(y))), test, 0)

X_train=X[train]
y_train=y[train]

X_test=X[test]
y_test=y[test]

count = 0
classifiers = []

print('============ Data Loaded ============')
print('X train shape: ' + str(X_train.shape))
print('y train shape: ' + str(y_train.shape))
print('Subject test: ' + str(subj+1))

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
            
df.to_excel('CV_round2_' + str(subj+1) + '.xlsx')

    

