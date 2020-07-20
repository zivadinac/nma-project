#@title Data retrieval
# from scipy.stats import zscore
# from sklearn.decomposition import PCA 
# from umap import UMAP
# from scipy.ndimage import uniform_filter1d
# from matplotlib import rcParams 
# rcParams['figure.figsize'] = [20, 4]
# rcParams['font.size'] =15
# rcParams['axes.spines.top'] = False
# rcParams['axes.spines.right'] = False
# rcParams['figure.autolayout'] = True

import os, requests
import numpy as np
from matplotlib import pyplot as plt


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

#try different alphas for more or less regularization
reg = linear_model.Lasso(alpha=0.1)#10)
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


# print(dat['sresp'].shape)
# print(dat['pupilArea'].shape)
# dat['pupilArea']

# predict pupil area by neuronal response 


#plt.hist(dat['pupilArea'].flatten(),bins=150)

