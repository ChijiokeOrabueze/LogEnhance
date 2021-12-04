

import numpy as np
import matplotlib.pyplot as plt
import copy, time, h5py

np.random.seed(2)

#DATA LOADING
with h5py.File('train_log_150.h5', 'r') as trainData:
    train_set_X_orig = np.array(trainData['Train_X'])
    train_set_Y_orig = np.array(trainData['Train_Y'])
    classes = np.array(trainData['classes'])


#NOMALIZATION
Train_X = train_set_X_orig.reshape(train_set_X_orig.shape[0],-1).T/255
Train_Y = copy.deepcopy(train_set_Y_orig)

