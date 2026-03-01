import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def binary_cross_entropy(p, y):
    n = p.shape[0]
    return -np.sum(
        y*np.log(p)+(1-y)*np.log(1-p)
    )/n

def weights_derivative(X, p, y):
    n = X.shape[0]
    return (np.transpose(X)@(p-y))/n

def bias_derivative(n, p, y):
    return np.sum(p-y)/n 

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    w = np.zeros((X.shape[1]),)
    b = 0
    X = np.array(X)
    y = np.array(y)
    for step in range(steps):
        p = _sigmoid(X@w+b)

        loss = binary_cross_entropy(p, y)
        dw = weights_derivative(X, p, y)
        db = bias_derivative(X.shape[0], p, y)
        w = w-(lr*dw)
        b = b-(lr*db)
    return (w,b)