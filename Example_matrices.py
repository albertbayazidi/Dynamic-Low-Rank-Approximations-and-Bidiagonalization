import numpy as np
from scipy.linalg import expm
from scipy.sparse import diags

def generateA(epsilon):
    """
    Generates A1 and A2 for the bigger A-matrix
    input: 
    epsilon - weighting variable 
    output:
    The matrix A1 or A2
    """
    first_block =  np.random.rand(10,10)/2 + np.identity(10)
    A = epsilon * np.random.rand(100,100)
    A [:10,:10] = first_block
    return A



def A(t, epsilon = 1/2): 
    """
    Generate A-matrix as described in task 4
    input: 
    t - time variable
    epsilon - weighting variable for A1 and A2
    output:
    the matrix A evaluated at time t
    """
    A1 = generateA(epsilon)
    A2 = generateA(epsilon)
    
    Id = np.identity(100)
    
    T1 = np.diag(-np.ones(99), k=1) + np.diag(np.ones(99), k=-1)
    T2 = np.diag(-np.ones(99)/2, k=1) + np.diag(-np.ones(98), k=2) + np.diag(np.ones(99)/2, k=-1) + np.diag(np.ones(98), k=-2)
    
    Q1 = lambda t :  Id * expm(t *T1)
    Q2 = lambda t :  Id * expm(t *T2)
    
    return Q1(t) @ (A1 + np.exp(t) * A2) @ Q2(t).T

def A_dot(t, epsilon = 1/2):
    """
    Generate the time-derivative of A-matrix as described in task 4
    input: 
    t - time variable
    epsilon - weighting variable for A1 and A2
    output:
    the derivative of matrix A evaluated at time t
    """

    A1 = generateA(epsilon)
    A2 = generateA(epsilon)
    
    Id = np.identity(100)
    
    T1 = np.diag(-np.ones(99), k=1) + np.diag(np.ones(99), k=-1)
    T2 = np.diag(-np.ones(99)/2, k=1) + np.diag(-np.ones(98), k=2) + np.diag(np.ones(99)/2, k=-1) + np.diag(np.ones(98), k=-2)
    
    Q1 = lambda t :  Id * expm(t *T1)
    Q2 = lambda t :  Id * expm(t *T2)
    
    # using chain rule
    return (T1 @ Q1(t) @ A1 + np.exp(t)*(Id + T1)@ Q1(t) @ A2) @ Q2(t).T + (Q1(t) @ A1 + np.exp(t) * Q1(t) @ A2) @ (T2@Q2(t)).T
