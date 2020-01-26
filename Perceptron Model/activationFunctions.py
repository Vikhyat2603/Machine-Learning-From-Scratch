import numpy as np

def ReXu(x, derivative=False):
    '''Non-asymptotic Rectified Exponential(exponent: 0.5) Unit
    View graph at: https://www.desmos.com/calculator/eyetqjpkxb'''
    if derivative:
        return (1+np.sign(x)+0.5*(1-np.sign(x))/((0.25+abs(x))**0.5))/2
    return ((1-np.sign(x))*(0.5-(0.25+abs(x))**0.5)+x+abs(x))/2

def atan_scaled(x, derivative=False):
    '''f(x) = np.arctan(x)/np.pi+0.5'''
    if derivative:
        return 1/(np.pi*(np.square(x)+1))
    return np.arctan(x)/np.pi+0.5
    
def atan(x, derivative=False):
    '''f(x) = arctan(x)'''
    if derivative:
        return 1/(np.square(x)+1)
    return np.arctan(x)

def leaky_ReLu(x, derivative=False):
    '''f(x) = (x+0.95*abs(x))/2'''
    if derivative:
        return (1+0.95*np.sign(x))/2
    return (x+0.95*np.abs(x))/2

def identity(x,derivative = False):
    '''f(x) = x'''
    return 1 if derivative else x

def ReLu(x,derivative = False):
    '''f(x) = (x+abs(x)) / 2'''
    if derivative:
        return (np.sign(x)+1)/2
    return (x+np.abs(x))/2

def sigmoid(x, derivative = False):
    '''f(x) = 1/(1+np.exp(-x))'''
    if derivative:
        return sigmoid(x)*(1-sigmoid(x))
    return 1/(1+np.exp(-x))

def createScaledFunction(func, alpha):
    ''' Get function with scale factor alpha : f(alpha*x)'''
    return (lambda x, derivative = False: alpha * func(alpha*x, True) if derivative else func(alpha*x))
    #d/dx [f(ax)] = a * f'(ax)
    
