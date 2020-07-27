import pickle
import numpy as np
import matplotlib.pyplot as plt
import movement 
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression 
from sklearn.decomposition import PCA
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt.plots import *


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

decoders = {}
train_acc = {}
test_acc = {}
def model(det_window, penalty, neuron_num=None, pca_com=None):
    if neuron_num is not None:
        neurons_idx = np.random.randint(0, len(neural_data), neuron_num)
        feat = extract_features(neural_data, neurons_idx=neurons_idx)
        key = neuron_num
    elif pca_com is not None:
        feat = extract_features(neural_data, pca_comp_num=pca_com)
        key = pca_com
    else:
        raise ValueError("Either `neuron_num` or `pca_com` must be provided.")

    X_train, y_train, X_test, y_test = prepare_data(feat, run_onset, det_window, 0.2)
    C = np.logspace(-8, 0, 50)

    decoder = LogisticRegressionCV(Cs=C, penalty=penalty, solver='liblinear', max_iter=100)
    decoder.fit(X_train, y_train)
    acc_test = decoder.score(X_test, y_test)
    acc_train = decoder.score(X_train, y_train)

    decoders[(key, det_window, penalty)] = decoder
    train_acc[(key, det_window, penalty)] = acc_train
    test_acc[(key, det_window, penalty)] = acc_test
    return np.abs(acc_train - acc_test) if acc_test > 0.7 else 100

def model_pca(args):
    pca_com, det_window, penalty = args
    return model(det_window, penalty, pca_com=pca_com)

def model_neurons(args):
    neuron_num, det_window, penalty = args
    return model(det_window, penalty, neuron_num=neuron_num)

# set seed
seed = np.random.seed(2020)

#IMPORT DATA
dat = np.load('data/stringer_spontaneous.npy', allow_pickle=True).item()
neural_data = dat['sresp']
run_data = dat['run']
run_onset, run_speed = movement.detect_movement_onset(run_data)

pca_com = Integer(low=1, high=100)
neuron_num = Integer(low=200, high=len(neural_data))
det_window = Integer(low=1, high=7)
penalty = Categorical(categories=["l1", "l2"], name="penalty")

use_pca = False
if use_pca:
    dimensions = [pca_com, det_window, penalty]
    default_hyperparams = [50, 3, "l1"]
    hp_model = model_pca
else:
    dimensions = [neuron_num, det_window, penalty]
    default_hyperparams = [200, 3, "l1"]
    hp_model = model_neurons

res = gp_minimize(func=hp_model, dimensions=dimensions, n_calls=20, x0=default_hyperparams)
opt_hp = tuple(res.x)

print(f"HP search done: {opt_hp}: {res.fun}. Train acc: {train_acc[opt_hp]}, test acc: {test_acc[opt_hp]}")
d = decoders[opt_hp]

with open(f"opt_hp_{'pca' if use_pca else 'neurons'}.pck", "wb") as opt_hp_f:
    pickle.dump(res, opt_hp_f, protocol=4)

