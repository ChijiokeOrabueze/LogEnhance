

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


#MODEL
def nn_model(X,Y,n_h,no_itr,learning_rate,print_cost):
    
    np.random.seed(3)
    n_x = layer_sizes(X,Y,n_h)[0]
    n_y = layer_sizes(X,Y,n_h)[2]
    costs = []
    
    parameters = initialize_params(n_x,n_h,n_y)
    
    for i in range(no_itr):
        A2,cache = Forward_Propagation(X,parameters)
        grads = Backward_Propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads,learning_rate)
        
        if i%100 == 0 or i == no_itr-1:
            cost = compute_cost(A2,Y)
            costs.append(cost)
            if print_cost:
                print(f'cost after {i} iterations = {cost}')
        
    return parameters
    
    
#MODEL TESTING AND ANALYSIS
model = nn_model(Train_X,Train_Y,6,1200,0.01,True)

#TEST DATA LOADING
with h5py.File('test_log_150.h5', 'r') as testData:
    test_X_orig = np.array(testData['test_x_orig'])
    test_X = np.array(testData['Test_X'])
    test_Y = np.array(testData['Test_Y'])


#TRAIN ANALYSIS
model = nn_model(Train_X,Train_Y,6,1200,0.01,True)