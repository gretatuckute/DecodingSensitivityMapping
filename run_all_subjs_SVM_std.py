# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:23:18 2018

@author: Greta, grtu@dtu.dk

#EXAMPLE FUNCTION CALL
#	python runSVM.py -s 0

"""

#Imports
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

#os.chdir('C:/Users/Greta/Documents/GitHub/decoding/')
from std import std_windows 

# Load data
#os.chdir('C:/Users/Greta/Documents/GitHub/decoding/data/ASR/')
ASR = sio.loadmat('ASRinterp')
X = ASR['EV_Z']
X = np.reshape(X,[10350,32,60])
X = std_windows(X,time_window=2)

y = ASR['Animate']
y = np.squeeze(y)

print('============ Data Loaded ============')
print('X shape: ' + str(X.shape))
print('y shape: ' + str(y.shape))


random_state = np.random.RandomState(0)

df = pd.DataFrame(columns=['Subject no.', 'scores_train', 'scores_test'])
cv = list(range(0,len(y),690))

count = 0

for counter, ii in enumerate(cv):
    
    classifier = SVC(random_state=random_state)
    test = list(range(ii, ii+690))
    train = np.delete(list(range(0, len(y))), test, 0)
    clf = classifier.fit(X[train], y[train])
    scores_train = clf.score(X[train], y[train])
    scores_test = clf.score(X[test], y[test])
    df.loc[count]=[counter+1, scores_train, scores_test]
    count += 1
    print(str(count))

            
df.to_csv('LOSO_std_' + str(date_str) + '.csv')

# pickle some variables
#filename = 'classifiers.pckl'
#pickle.dump(classifiers, open(filename, 'wb'))
    

