import os, requests
import numpy as np
from matplotlib import pyplot as plt
import time

# fname = "stringer_spontaneous.npy"
# url = "https://osf.io/dpqaj/download"

# if not os.path.isfile(fname):
#   try:
#     r = requests.get(url)
#   except requests.ConnectionError:
#     print("!!! Failed to download data !!!")
#   else:
#     if r.status_code != requests.codes.ok:
#       print("!!! Failed to download data !!!")
#     else:
#       with open(fname, "wb") as fid:
#         fid.write(r.content)

# #@title Data loading
dat = np.load('./data/stringer_spontaneous.npy', allow_pickle=True).item()


# Perform PCA on neural data
def pca_projection(data,var_threshold=0.9):
    """
    Parameters:
        data           -> expects matrix with shape (samples,features)
        var_threshold  -> percentage of explained variance; default is 0.9
    Returns:
        data projected on k principal components explaining var_threshold percentage of variance
    """
    #calculate covariance matrix
    n_samples = data.shape[0] # number of samples
    data = data - data.mean(axis=0) # normalize the data
    covMat =  (data.T @ data) / n_samples # compute the covariance matrix

    # Calculate Principal Components
    values, vectors = np.linalg.eig(covMat) # find eigenvectors
    var_explained = np.cumsum(values)/np.sum(values) # calculate variance explained
    k = np.sum(var_explained<var_threshold) # number of significant PCs

    scores_allPCs = data @ vectors # calculate the scores = the projection of the original data onto the new basis
    vectors_allPCs = vectors # eigenvectors
    scores_sigPCs = scores_allPCs[:,:k] # subset of scores for significant principal components
    vectors_sigPCs = vectors_allPCs[:,:k] # subset of eigenvectors for significant principal components
    
    return scores_sigPCs, vectors_sigPCs, scores_allPCs, vectors_allPCs, var_explained, var_threshold

# If PCA takes too long reduce number of neurons
num_neurons = 100 #set how many neurons to use
data_subset = dat['sresp'][:num_neurons,:].T # take a subset of the data
scores_sigPCs, vectors_sigPCs, scores_allPCs, vectors_allPCs, var_explained, var_threshold = pca_projection(data_subset) # run PCA on the designated subset of data
reconstructed_sigPCs =  (scores_sigPCs @ vectors_sigPCs.T) + data_subset.mean(axis=0)
reconstructed_allPCs =  (scores_allPCs @ vectors_allPCs.T) + data_subset.mean(axis=0)

plt.ylabel('Explained Variance')
plt.xlabel('Number of Principal Components')
plt.plot(var_explained)
plt.axhline(var_threshold,color='r')

plt.show()