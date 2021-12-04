import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import copy, time



train_data = h5py.File('train_log_150.h5', 'r')
train_set_x_orig = np.array(train_data['Train_X'])
train_set_y = np.array(train_data['Train_Y'], dtype = int)
data_classes = np.array(train_data['classes'])
train_data.close()