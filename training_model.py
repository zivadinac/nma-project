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

#%% LOGISTIC REGRESSION  - train model
# SET PARAMETERS
C = np.logspace(-4, 0, 20)
det_window = 3
neuron_num = 400
pca_com = 200

#use only one of this two lines!
feat = extract_features(neural_data, neurons_idx=np.random.randint(0, len(neural_data), neuron_num))
#feat = extract_features(neural_data, pca_comp_num =pca_com)

X_train, y_train, X_test, y_test = prepare_data(feat, run_onset, det_window, 0.2)

decoders = {}
train_acc = {}
test_acc = {}
for penalty in ['l1', 'l2']:
    decoder = LogisticRegressionCV(Cs = C, penalty=penalty, solver='liblinear')
    decoder.fit(X_train, y_train)
    acc_test = decoder.score(X_test,y_test)
    acc_train = decoder.score(X_train,y_train)
    decoders[penalty] = decoder
    test_acc[penalty] = acc_test
    train_acc[penalty] = acc_train

