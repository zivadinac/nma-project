#@title Data retrieval
import os, requests
import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA 
from umap import UMAP
from scipy.ndimage import uniform_filter1d
from matplotlib import rcParams 
from matplotlib import pyplot as plt
rcParams['figure.figsize'] = [20, 4]
rcParams['font.size'] =15
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['figure.autolayout'] = True


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
print(dat.keys())


from sklearn import linear_model


reg = linear_model.Lasso(alpha=0.1)#0.1)
# without bias ...
#reg.fit(dat['sresp'].T,dat['run'])
#with bias
X = np.ones(dat['sresp'].shape+np.array([1,0]))
X[1::,:]=dat['sresp']
reg.fit(X.T,dat['run'])


plt.plot(reg.coef_.flatten())
np.sum(reg.coef_.flatten()!=0)
plt.show()
pred = np.zeros(7018)
for i in range(7018):
    pred[i] = reg.predict([X[:,i]])

plt.plot(pred,dat['run'],'*')







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

# 

# print(dat['sresp'].shape)
# print(dat['pupilArea'].shape)

# dat['pupilArea']

# predict pupil area by neuronal response 

# i believe pupil data will be normally distributed
#plt.hist(dat['pupilArea'].flatten(),bins=150)



# movement = dat['run']
# plt.plot(movement>5)
# print(max(movement))

# how is resp distributed?
# make a histogram of neuronal responses
# what are the bins? smth. like number of spikes
# we have more neurons than samples! 

# plt.hist(dat['sresp'].flatten(),bins=150)
# plt.xlim(0,500)


# pop_mean = np.mean(dat['sresp'],axis=0)

# print(dat['sresp'].shape)

# print(dat['run'].shape)
# plt.plot(dat['run'],pop_mean,'*')
# plt.xlabel('speed')
# plt.ylabel('population_mean')
# #@title Basic properties of behavioral data using plot and scatter
# ax = plt.subplot(1,5,1)
# plt.plot(dat['pupilArea'][:500,0])
# ax.set(xlabel='timepoints', ylabel = 'pupil area')

# ax = plt.subplot(1,5,2)
# plt.plot(dat['pupilCOM'][:500,:])
# ax.set(xlabel='timepoints', ylabel = 'pupil XY position')

# ax = plt.subplot(1,5,3)
# plt.plot(dat['beh_svd_time'][:500,0])
# ax.set(xlabel='timepoints', ylabel = 'face SVD #0')

# ax = plt.subplot(1,5,4)
# plt.plot(dat['beh_svd_time'][:500,1])
# ax.set(xlabel='timepoints', ylabel = 'face SVD #1')

# ax = plt.subplot(1,5,5)
# plt.scatter(dat['beh_svd_time'][:,0], dat['beh_svd_time'][:,1], s = 1)
# ax.set(xlabel='face SVD #0', ylabel = 'face SVD #1')

# plt.show()


# #@title take PCA after preparing data by z-score
# Z = zscore(dat['sresp'], axis=1)
# Z = np.nan_to_num(Z)
# X = PCA(n_components = 200).fit_transform(Z)

# #@title run a manifold embedding algorithm (UMAP) in two or three dimensions. 
# ncomp = 1 # try 2, then try 3
# xinit = 1 * zscore(X[:,:ncomp], axis=0)
# embed = UMAP(n_components=ncomp, init =  xinit, n_neighbors = 20, 
#              metric = 'correlation', transform_seed = 42).fit_transform(X)


# #@title Plot PCs. Too many points, switch to logarithmic hexbin! 
# ax = plt.subplot(1,5,1)
# plt.scatter(X[:,0], X[:,1], s = 4, alpha = .1)
# ax.set(xlabel = 'PC 0 ', ylabel = 'PC 1')
# ax = plt.subplot(1,5,2)
# plt.hexbin(X[:,0], X[:,1], gridsize = 40, bins = 'log')
# ax.set(xlabel = 'PC 0 ', ylabel = 'PC 1', alpha = .1);

# embed = embed.flatten()
# isort = np.argsort(embed)
# RasterMap = uniform_filter1d(Z[isort, :], size= 50, axis=0)
# RasterMap = zscore(RasterMap[::10, :], axis = 1)

# plt.figure(figsize=(16,8))
# ax = plt.subplot(111)
# trange = np.arange(1100, 1400)
# plt.imshow(RasterMap[:, trange], vmax= 3, vmin = -1, aspect = 'auto', cmap = 'magma')
# ax.set(xlabel = 'timepoints', ylabel = 'sorted neurons');
