def sigmoid(z):
    
    s = 1/(1+np.exp(-z))
    return s


def layer_sizes(X,Y,n_h): #no of units in input, hidden and output layer respectively
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    return n_x, n_h, n_y


def initialize_params(n_x,n_h,n_y): #initializing W1,b1,W2,b2: weights and bias of input and hidden layers
    """
    Arguments:
        n_x = size of input layer
        n_h = size of hidden layer
        n_y = size of output layer
        
    Returns:
    Params --- A python dictionary containing the initialized weights:
        W1 = Weight matrix of shape (n_x,n_h)
        b1 = Bias vector of shape (n_h,1)
        W2 = Weight matrix of shape (n_h,n_y)
        b1 = Bias vector of shape (n_y,1)
    """
    np.random.seed(2)
    
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))
    
    params = {'W1':W1, 
              'b1':b1, 
              'W2':W2, 
              'b2':b2}
    
    return params

def Forward_Propagation(X,parameters):
    """
    Forward propagation fuction
    Arguments:
        X:array of input data
        parameters: python dictionary containing the learning paramaters
        
    Returns:
        A2: array containing the second activation
        Cache: A python dictionary containing 'Z1', 'A1', 'Z2' and 'A2'
    """
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {'Z1':Z1, 
             'A1':A1, 
             'Z2':Z2, 
             'A2':A2}
    
    return A2, cache


def compute_cost(A2, Y):
    """
    Computes the cross-entropy cost
    
    Arguments:
        A2: sigmoid output of second activtion with shape (1, no of training examples)
        Y: true labels of the input with shape (1, no of training examples)
        
    Returns:
        cost: the cross entropy cost
    
    """
    
    m = Y.shape[1]
    
    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),(1-Y))
    cost = (-1/m)*np.squeeze(np.sum(logprobs))
    
    return cost
    


def Backward_Propagation(parameters, cache, X, Y):
    """
    Arguments:
        parameters: python dictionary containing the learning paramaters
        Cache: A python dictionary containing 'Z1', 'A1', 'Z2' and 'A2'
        X:array of input data
        Y: true labels of the input with shape (1, no of training examples)
        
    Returns:
        grads: dictionary object containing the gradients of the weights and bias:
            'dW2', 'db2', 'dW1', 'db1'
    
    """
    
    m = X.shape[1]
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    A1 = cache['A1']
    A2 = cache['A2']
    
    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)
    
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A2,2))
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis = 1, keepdims = True)
    
    grads={'dW2':dW2, 
          'db2':db2, 
          'dW1':dW1, 
          'db1':db1}
    
    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Updates the parameters using the gradient descent rule: W = W-dW*learning rate
    
    Arguments:
        parameters: python dictionary containing the learning paramaters
        grads: dictionary object containing the gradients of the weights and bias:
            'dW2', 'db2', 'dW1', 'db1'
        learning_rate: integer value denoting the learning rate of the machine (rate of gradient descent)
        
    Returns:
        up_params: dictionary object containing the updated weights and bias values
    
    """
    
    W1 = copy.deepcopy(parameters['W1'])
    b1 = parameters['b1']
    W2 = copy.deepcopy(parameters['W2'])
    b2 = parameters['b2']
    
    
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    W1-=dW1*learning_rate
    b1-=db1*learning_rate
    W2-=dW2*learning_rate
    b2-=db2*learning_rate
    
    up_params = {'W1':W1, 
                 'b1':b1, 
                 'W2':W2, 
                 'b2':b2}
    
    return up_params


