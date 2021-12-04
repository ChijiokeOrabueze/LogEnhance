import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import copy, time



def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s


def initialize_param(dim):
    w = np.zeros((dim,1))
    b = 0.00
    
    params = {'w':w,
             'b':b}
    
    return params



def propagate(X,Y,w,b):
    m = X.shape[1]
    
    z = np.dot(w.T,X) + b
    a = sigmoid(z)
    
    cost = (-1/m)*np.sum((Y*np.log(a)) + ((1-Y)*(np.log(1-a))))
    
    dz = a - Y
    dw = np.dot(X,dz.T)/m
    db = np.sum(dz)/m
    
    cost = np.squeeze(cost)
    
    grads = {'dz':dz,
             'dw':dw,
             'db':db}
    
    return cost, grads



def optimize(X,Y,w,b,no_itr,learn_rate,show_cost=True):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []
    for i in range(no_itr):
        cost, grads = propagate(X,Y,w,b)
        
        dw = grads['dw']
        db = grads['db']
        
        w-=learn_rate*dw
        b-=learn_rate*db
        
        if show_cost:
            if i%100 == 0:
                costs.append(cost)
                print(f'cost after {i} iterations = {cost}')
                
    L_params = {'w':w,
               'b':b}
    
    cur_grads = {"dw":dw,
                 "db":db}
    
    return L_params, cur_grads, costs


def predict(X,W,b):
    m = X.shape[1]
    z = np.dot(W.T,X) + b
    A = sigmoid(z)
    
    y_predictions = np.zeros((1,m))
    
    for i in range(A.shape[1]):
        if A[0,i] > 0.5:
            y_predictions[0,i] = 1
        else:
            y_predictions[0,i] = 0
            
    return y_predictions



