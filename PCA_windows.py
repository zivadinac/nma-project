# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:49:41 2020

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import movement 
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression 
from sklearn.decomposition import PCA


def extract_features(neural_data, neurons_idx=None, pca_comp_num=None):
    
    if neurons_idx is None and pca_comp_num is None:
        return neural_data

    if neurons_idx is not None:
        return neural_data[neurons_idx, :]
 
    if pca_comp_num is not None:
        pca = PCA(pca_comp_num)
        pca.fit(neural_data.T)
        return pca.transform(neural_data.T).T

# set seed
seed = np.random.seed(2020)

# #IMPORT DATA
dat = np.load('data/stringer_spontaneous.npy', allow_pickle=True).item()
neural_data = dat['sresp']
run_data = dat['run']
run_onset, run_speed = movement.detect_movement_onset(run_data)

#%% LOGISTIC REGRESSION ON NEURAL DATA - train model
# SET PARAMETERS
C = np.logspace(-4, 0, 20)
det_window = 13
neuron_num = 4000
pca_com = 1

# CORR of PCA in windows
# window_corr = np.zeros([(neural_data.shape[1]-(det_window+1)),1])
window_corr = []

for idx in range(det_window+1, len(neural_data[0,:])):
    neural_data_window = neural_data[:,(idx-det_window):idx]
    feat_w = extract_features(neural_data_window, pca_comp_num =pca_com)
    run_w = run_data[(idx-det_window):idx]
    window_corr.append(np.corrcoef(feat_w.flatten(), run_w.flatten()))

plt.plot(window_corr, label= 'corr 1PC-run speed')
plt.plot(run_onset[det_window+1:]*np.max(window_corr), 'or', label = 'run onset')
plt.xlim(0,1000)
plt.title('correlation 1PC with run velocity')
plt.legend()

# CORR of PCA of all neural vector
pca_com = 2000
feat = extract_features(neural_data, pca_comp_num =pca_com)

corr_PC_run = []
for PC in range(pca_com):
    corr_PC_run.append(np.corrcoef(feat[PC,:].flatten(),run_data.flatten())[0])

plt.plot(corr_PC_run, label= 'corr PCs - run speed')
plt.plot(run_onset[det_window+1:]*np.max(window_corr), 'or')
plt.xlim(0,1000)


#use only one of this two lines!
# feat = extract_features(neural_data, pca_comp_num =pca_com)