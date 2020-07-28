# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 23:06:37 2020

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import movement 
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression 
from sklearn.decomposition import PCA


def prepare_data(neural_data, run_onset, det_window, perc_test):
    
    ''' Prepare data from Stringer dataset to Neuroduck GLM. 
    
    Parameters
    ----------
    neural_data : neurons x timepoints array
    run_onset : binary matrix indicating onset of 'run'
    det_window : single value of the window in timebins

    Returns
    -------
    features : np.array of size number of windows x feature length
    feat_labels : 0 for no-run-onset, 1 for run-onset 

    '''
    run_onset[:det_window+1]=0
    
    features = np.zeros((2*run_onset.sum(), len(neural_data)))
    
    no_onset = np.zeros_like(run_onset)
    no_onset[np.random.randint(det_window+1, len(run_onset), run_onset.sum())]=1
    
    count = 0
    for label in [run_onset, no_onset]:
        for event_idx, event in enumerate(label):
            if event ==1:         
                neural_act = neural_data[:, event_idx - det_window : event_idx]
                neural_act = neural_act.mean(axis =1).T
                features[count, :] = neural_act
                count += 1
    
    feat_labels = np.zeros(2*run_onset.sum())
    feat_labels[0:run_onset.sum()] = 1
        
    shuffle_idx = np.random.permutation(len(feat_labels))
    
    n_test = int(perc_test * len(features))
    
    test_idx_0 = shuffle_idx[np.where(feat_labels == 0)][0:n_test // 2]
    train_idx_0 = shuffle_idx[np.where(feat_labels == 0)][n_test // 2:]
    test_idx_1 = shuffle_idx[np.where(feat_labels == 1)][0:n_test // 2]
    train_idx_1 = shuffle_idx[np.where(feat_labels == 1)][n_test // 2:]
 
    train_features = features[np.hstack([train_idx_0, train_idx_1]), :]
    train_labels = feat_labels[np.hstack([train_idx_0, train_idx_1])]
  
    test_features = features[np.hstack([test_idx_0, test_idx_1]), :]
    test_labels = feat_labels[np.hstack([test_idx_0, test_idx_1])]
    
    return train_features, train_labels, test_features, test_labels         



def prepare_data_delay(neural_data, run_onset, det_window, delay, perc_test=.2):
    seed = np.random.seed(2020)

    ''' Prepare data from Stringer dataset to Neuroduck GLM. 
    
    Parameters
    ----------
    neural_data : neurons x timepoints array
    run_onset : binary matrix indicating onset of 'run'
    det_window : single value of the window in timebins
    delay: single value of timebins delay
    
    Returns
    -------
    features : np.array of size number of windows x feature length
    feat_labels : 0 for no-run-onset, 1 for run-onset 

    '''
    run_onset[:det_window+1+delay]=0
    
    features = np.zeros((2*run_onset.sum(), len(neural_data)))
    
    no_onset = np.zeros_like(run_onset)
    no_onset[np.random.randint(det_window+1+delay, len(run_onset), run_onset.sum())]=1
    
    count = 0
    for label in [run_onset, no_onset]:
        for event_idx, event in enumerate(label):
            if event ==1:         
                neural_act = neural_data[:, event_idx - det_window -delay: event_idx - delay]
                neural_act = neural_act.mean(axis =1).T
                features[count, :] = neural_act
                count += 1
    
    feat_labels = np.zeros(2*run_onset.sum())
    feat_labels[0:run_onset.sum()] = 1
        
    shuffle_idx = np.random.permutation(len(feat_labels))
    
    n_test = int(perc_test * len(features))
    
    test_idx_0 = shuffle_idx[np.where(feat_labels == 0)][0:n_test // 2]
    train_idx_0 = shuffle_idx[np.where(feat_labels == 0)][n_test // 2:]
    test_idx_1 = shuffle_idx[np.where(feat_labels == 1)][0:n_test // 2]
    train_idx_1 = shuffle_idx[np.where(feat_labels == 1)][n_test // 2:]
 
    train_features = features[np.hstack([train_idx_0, train_idx_1]), :]
    train_labels = feat_labels[np.hstack([train_idx_0, train_idx_1])]
  
    test_features = features[np.hstack([test_idx_0, test_idx_1]), :]
    test_labels = feat_labels[np.hstack([test_idx_0, test_idx_1])]
    
    return train_features, train_labels, test_features, test_labels         
    

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
det_window = 1
neuron_num = 4000
pca_com = 13

#use only one of this two lines!
feat = extract_features(neural_data, neurons_idx=np.random.randint(0, len(neural_data), neuron_num))
# feat = extract_features(neural_data, pca_comp_num =pca_com)

delay_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

test_acc = np.zeros([len(delay_bins), 2])
train_acc = np.zeros([len(delay_bins), 2])


for delay in range(0,len(delay_bins)):
    X_train, y_train, X_test, y_test = prepare_data_delay(feat, run_onset, det_window, delay, 0.2)
    
    decoder = LogisticRegressionCV(Cs = C, penalty="l1", solver='liblinear')
    decoder.fit(X_train, y_train)
    acc_test = decoder.score(X_test,y_test)
    acc_train = decoder.score(X_train,y_train)
    test_acc[delay, 0] = acc_test
    train_acc[delay,0] = acc_train
    
    decoder = LogisticRegressionCV(Cs = C, penalty="l2", solver='liblinear')
    decoder.fit(X_train, y_train)
    acc_test = decoder.score(X_test,y_test)
    acc_train = decoder.score(X_train,y_train)
    test_acc[delay, 1] = acc_test
    train_acc[delay,1] = acc_train

# PLOT
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))
                      
ax1.plot(delay_bins, test_acc[:,0].flatten(), 'b.-', label='test acc')
ax1.plot(delay_bins, train_acc[:, 0].flatten(), 'r.-', label='train acc')
ax1.set(title = "l1 regularization (f rate)", xlabel= "delay(timebins)")
ax1.legend()
                
ax2.plot(delay_bins, test_acc[:,1], 'b.-', label='test acc')
ax2.plot(delay_bins, train_acc[:, 1], 'r.-', label='train acc')
ax2.set(title= "l2 regularization (f rate)", xlabel= "delay(timebins)")
ax2.legend()
                  
fig.suptitle (f'frate-GLM on {neuron_num} neurons with a time window of {det_window} timebins')


#%% GLM ON MESSY NEURAL DATA
neural_data_messy = neural_data[:,np.random.permutation(len(neural_data[0,:]))]

# SET PARAMETERS
C = np.logspace(-4, 0, 20)
det_window =5
neuron_num = 2000
pca_com = 200

#use only one of this two lines!

#use only one of this two lines!
feat = extract_features(neural_data_messy, neurons_idx=np.random.randint(0, len(neural_data_messy), neuron_num))
#feat = extract_features(neural_data, pca_comp_num =pca_com)

delay_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

test_acc = np.zeros([len(delay_bins), 2])
train_acc = np.zeros([len(delay_bins), 2])

for delay in range(0,len(delay_bins)):
    X_train, y_train, X_test, y_test = prepare_data_delay(feat, run_onset, det_window, delay, 0.2)
    
    decoder = LogisticRegressionCV(Cs = C, penalty="l1", solver='liblinear')
    decoder.fit(X_train, y_train)
    acc_test = decoder.score(X_test,y_test)
    acc_train = decoder.score(X_train,y_train)
    test_acc[delay, 0] = acc_test
    train_acc[delay,0] = acc_train
    
    decoder = LogisticRegressionCV(Cs = C, penalty="l2", solver='liblinear')
    decoder.fit(X_train, y_train)
    acc_test = decoder.score(X_test,y_test)
    acc_train = decoder.score(X_train,y_train)
    test_acc[delay, 1] = acc_test
    train_acc[delay,1] = acc_train

# PLOT
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))
                      
ax1.plot(delay_bins, test_acc[:,0].flatten(), 'b.-', label='test acc')
ax1.plot(delay_bins, train_acc[:, 0].flatten(), 'r.-', label='train acc')
ax1.set(title = "l1 regularization (f rate)", xlabel= "delay(timebins)")
ax1.legend()
                
ax2.plot(delay_bins, test_acc[:,1], 'b.-', label='test acc')
ax2.plot(delay_bins, train_acc[:, 1], 'r.-', label='train acc')
ax2.set(title= "l2 regularization (f rate)", xlabel= "delay(timebins)")
ax2.legend()

fig.suptitle (f'MESSY DATA: frate-GLM on {neuron_num} neurons with a time window of {det_window} timebins')

