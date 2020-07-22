import os, requests
import numpy as np
from matplotlib import pyplot as plt
import time

fname = "stringer_spontaneous.npy"
url = "https://osf.io/dpqaj/download"

if not os.path.isfile(fname):
  try:
    r = requests.get(url)
  except requests.ConnectionError:
    print("!!! Failed to download data !!!")
  else:
    if r.status_code != requests.codes.ok:
      print("!!! Failed to download data !!!")
    else:
      with open(fname, "wb") as fid:
        fid.write(r.content)

#@title Data loading
dat = np.load('stringer_spontaneous.npy', allow_pickle=True).item()

# Perform PCA on neural data



# Calculate Covariance Matrix
# If PCA takes too long reduce number of neurons
start = time.time()
num_neurons = 100
Neurons = dat['sresp'][:num_neurons,:] #shape (11983, 7018)
Neurons = Neurons.T
Neurons_Mean =  np.mean(Neurons,axis=0)
Neurons = Neurons - Neurons_Mean
N_samples = Neurons.shape[0]
X = (1/N_samples) * Neurons.T @ Neurons   #shape(11983,11983)

# Calculate Principal Components 
values, vectors = np.linalg.eig(X) #eig -> eigenvalues and vectors in descending order
# values, vectors = np.linalg.eigh(X) #eigh -> eigenvalues and vectors in ascending order                              
# values = values[::-1]
# vectors = vectors[:,::-1]

end = time.time()
print(end - start)

#Take the subset of eigenvalues and eigenvectors that explain x percent of the variance
var_explained = np.cumsum(values)/np.sum(values)
threshold = 0.8
values_ = values[var_explained<threshold]
vectors_ = vectors[var_explained<threshold]
K = np.sum(var_explained<threshold)

plt.ylabel('Explained Variance')
plt.xlabel('Number of Principal Components')
plt.plot(var_explained)
plt.axhline(threshold,color='r')

plt.show()
K=100

# Project Neural Data from Standard Basis into new Basis that is the Eigenvectors of the Covariance Matrix
score = Neurons @ vectors
Neurons_reconstructed =  (score[:, :K] @ vectors[:, :K].T) + Neurons_Mean


#projection_matrix = (eigen_vectors.T[:][:2]).T

#from sklearn.decomposition import PCA
# Neurons = dat['sresp'] #shape (11983, 7018)
# Neurons = Neurons - np.mean(Neurons,axis=0)
# pca = PCA(11983)
# pca.fit(Neurons)
# sort pupil position into 4 distinct categories

# nan_removed = dat['pupilCOM'][~np.isnan(dat['pupilCOM'].any(axis=1))]
# np.where(np.logical_or(np.isnan(dat['pupilCOM'][:,0]),np.isnan(dat['pupilCOM'][:,1])))
# x_median = np.median(nan_removed[:,0])
# y_median = np.median(nan_removed[:,1])

# plt.scatter(dat['pupilCOM'][:,0]>x_median,dat['pupilCOM'][:,1]>y_median)

# from sklearn import linear_model
# # setup for a linear model 
# # try different alphas for more or less regularization
# reg = linear_model.Lasso(alpha=10)#10)
# # use a bias term (intercept)
# X = np.ones(dat['sresp'].shape+np.array([1,0]))
# X[1::,:] = dat['sresp']
# # fit neural data to movement speed
# reg.fit(X.T,dat['run'])
# # plot coefficients on neurons
# plt.plot(reg.coef_[1::].flatten())
# # get number of nonzero coefficients
# print(np.sum(reg.coef_.flatten()!=0))
# plt.show()
# # predict the movement speed by neural data 
# # (beware no differentiation between training and testset)
# pred = np.zeros(7018)
# for i in range(7018):
#     pred[i] = reg.predict([X[:,i]])

# plt.plot(pred,dat['run'],'*')



# from scipy.signal import find_peaks
# dat_pupilA = dat['pupilArea']
# peaks, _ = find_peaks(dat_pupilA.flatten(), [500,2500])
# print(peaks)
# peaks.shape
# plt.plot(dat_pupilA)

### testing/exploring
# plt.plot(np.arange(7018),np.mean(dat['sresp'],axis=0))
# plt.show()

# print(dat['pupilCOM'].shape)
# #plt.plot(dat['pupilCOM'][:,0],dat['pupilCOM'][:,1],'.')

# grad = np.diff(dat['pupilCOM'],axis=0)

# #x = np.logical_or(grad[:,0]>5, grad[:,1]>5)
# plt.plot(np.mean(dat['sresp'],axis=0)[1::],np.abs(grad[:,0])+np.abs(grad[:,1]),'.')
# plt.show()

#plt.plot(np.mean(dat['sresp'],axis=1),x)
# for i in range(7018):
#     plt.plot(dat['pupilCOM'][i,0],dat['pupilCOM'][i,1],'b*')
#     plt.draw()
#     plt.pause(0.1)

# print(dat['sresp'].shape)
# print(dat['pupilArea'].shape)
# dat['pupilArea']

# predict pupil area by neuronal response 

#plt.hist(dat['pupilArea'].flatten(),bins=150)


# Perform PCA on neural data

# def pca_projection(data,var_threshold=0.9):
#     """
#     Args:
#         data           -> expects matrix with shape (samples,features)
#         var_threshold  -> percentage of explained variance 
#     Returns:
#         data projected on k principal components explaining var_threshold percentage of variance
#     """

#     n_samples = data.shape[0]
#     data = data - data.mean(axis=0)
#     cov =  data.T @ data / n_samples
    
#     values, vectors = np.linalg.eig(X)
    
#     var_explained = np.cumsum(values)/np.sum(values)
#     k = np.sum(var_explained<var_threshold)
    
#     values = values[var_explained<var_threshold]
#     vectors = vectors[var_explained<var_threshold]
    
#     print(values.shape)
#     print(vectors.shape)
#     score = data @ vectors
    
#     return score