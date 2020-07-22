# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:45:30 2020

@author: Neuroducks
"""
import numpy as np
import matplotlib.pyplot as plt
import movement 
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_val_score,  train_test_split, ShuffleSplit

from sklearn.metrics import accuracy_score

# #IMPORT DATA
dat = np.load('data/stringer_spontaneous.npy', allow_pickle=True).item()
neural_data = dat['sresp']
run_data = dat['run']
run_onset, run_speed = movement.detect_movement_onset(run_data)

# SET PARAMETERS
det_window = 3

def subset_from_data (neural_data, n, seed = 2020):

    ''' Sample n random columns from the input matrix
        Parameters
        ----------
        neural_data : np.array of size neurons x timepoints 
        n: sample size (integer) 
        
        Returns
        -------
        neural_data_subset: np.array of size n x timepoits
    '''
    np.random.seed (seed)
    neural_data_subset = neural_data[np.random.randint(0, len(neural_data), size = n)]
    
    return neural_data_subset

def prepare_data(neural_data, run_onset, det_window):
    
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
    features = features[shuffle_idx, :]
    feat_labels = feat_labels[shuffle_idx]
    
    return features, feat_labels         
            
X, y = prepare_data(neural_data, run_onset, det_window)

#%% LOGISTIC REGRESSION without regularization
decoder = LogisticRegressionCV()
decoder.fit(X, y)
acc = decoder.score(X,y)

#%% SET APPROPIATE HYPERPARAMETERS FOR REGULARIZATION (from NMA google colaborate)

def model_selection(X, y, C_values):
  """Compute CV accuracy for each C value.

  Args:
    X (2D array): Data matrix
    y (1D array): Label vector
    C_values (1D array): Array of hyperparameter values

  Returns:
    accuracies (1D array): CV accuracy with each value of C

  """
  accuracies = []
  for C in C_values:

    # Initialize and fit the model
    # (Hint, you may need to set max_iter)
    model = LogisticRegression(penalty="l2", C=C, max_iter=1000)


    # Get the accuracy for each test split
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    accs = cross_val_score(model, X, y, cv=cv)

    # Store the average test accuracy for this value of C
    accuracies.append(accs.mean())

  return accuracies

def plot_model_selection(C_values, accuracies):
  """Plot the accuracy curve over log-spaced C values."""
  ax = plt.figure().subplots()
  ax.set_xscale("log")
  ax.plot(C_values, accuracies, marker="o")
  best_C = C_values[np.argmax(accuracies)]
  ax.set(
      xticks=C_values,
      xlabel="$C$",
      ylabel="Cross-validated accuracy",
      title=f"Best C: {best_C:1g} ({np.max(accuracies):.2%})",
      )


C_values = np.logspace(-4, 4, 9) # Use log-spaced values for C

accuracies = model_selection(X, y, C_values)
plot_model_selection(C_values, accuracies)