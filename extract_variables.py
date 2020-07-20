# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:29:12 2020

@author: Neuroducks
"""

#%% LOAD VARIABLES

dat_srate = dat['sresp'] # y
dat_run = dat['run']
dat_pupilA = dat['pupilArea']

#%% DICHOTOMIZE dat_run
thres = 8
dat_run2 = np.zeros_like (dat_run)

idx = dat_run> thres

# COUNT JUST INITIATION OF MOVEMENT
dat_run2 = np.zeros_like (dat_run)
for i in range(1,len(dat_run)-1):
    if idx[i-1]==1: 
        dat_run2[i] = 0
    else: dat_run2[i] = idx[i]
