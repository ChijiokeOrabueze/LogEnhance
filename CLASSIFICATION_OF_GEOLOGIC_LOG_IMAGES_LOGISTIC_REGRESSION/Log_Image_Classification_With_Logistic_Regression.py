import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import copy, time
from HelperFunctions import sigmoid, initialize_param, propagate, optimize, predict


#DATA LOADING
train_data = h5py.File('../train_log_150.h5', 'r')
train_set_x_orig = np.array(train_data['Train_X'])
train_set_y = np.array(train_data['Train_Y'], dtype = int)
data_classes = np.array(train_data['classes'])
train_data.close()


#DATA VISUALIZATION
index = 27
plt.imshow(train_set_x_orig[index])
print(f'y = {int(np.squeeze(train_set_y[:,index]))}, This Image is {data_classes[int(np.squeeze(train_set_y[:,index]))].decode("utf-8")} ')


#NOMALIZATION OF IMAGE DATA
flat_train_x = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
train_X = flat_train_x/255
train_Y = train_set_y

#LINEAR REGRESSION MODEL
def ln_model(X_train,Y_train,X_test,Y_test,no_itr,learn_rate):
    dim = X_train.shape[0]
    params = initialize_param(dim)
    
    w = params['w']
    b = params['b']
    
    L_params, cur_grads, costs = optimize(X_train,Y_train,w,b,no_itr,learn_rate)
    
    W_L = L_params['w']
    b_L = L_params['b']
    
    train_predictions = predict(X_train,W_L,b_L)
    test_predictions = predict(X_test,W_L,b_L)
    
    print(f'train acuracy = {100-np.mean(np.abs(train_predictions - Y_train))*100}%')
    print(f'test accuracy = {100 - np.mean(np.abs(test_predictions - Y_test))*100}%')
    
    d = {'costs':costs, 
         'train_predictions':train_predictions,
        'test_predictions':test_predictions,
        'W':W_L,
        'b':b_L,
        'iterations':no_itr,
        'learn_rate':learn_rate,
        'cur_grads':cur_grads}
    
    return d


#MODEL ANALYSIS

#TEST DATA LOADING
with h5py.File('../test_log_150.h5', 'r') as testData:
    test_X_orig = np.array(testData['test_x_orig'])
    X_test = np.array(testData['Test_X'])
    Y_test = np.array(testData['Test_Y'])

#CREATING MODEL
model = ln_model(train_X,train_Y,X_test,Y_test,1000,0.00001)

#ANALYSIS
costs=np.squeeze(model['costs'])
plt.plot(costs)
plt.xlabel('number of iterations per 100')
plt.ylabel('costs')
plt.title('learning rate = ' + str(model['learn_rate']) + '\n' + 'number of iteration = '+ str(model['iterations']))
plt.show()