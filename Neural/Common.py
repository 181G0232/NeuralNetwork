import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivate(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def tanh(x):
    return np.tanh(x)
 
def tanh_derivate(x):
    return np.arccos(x)**2