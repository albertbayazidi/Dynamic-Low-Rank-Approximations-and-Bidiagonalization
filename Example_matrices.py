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

# Generate A1 and A2 once, to be used in all experiments
np.random.seed(1)
A1 = generateA(0.1)
np.random.seed(2)
A2 = generateA(0.1)

def A(t): 
    """
    Generate A-matrix as described in task 4
    input: 
    t - time variable
    epsilon - weighting variable for A1 and A2
    output:
    the matrix A evaluated at time t
    """
    
    Id = np.identity(100)
    
    T1 = np.diag(-np.ones(99), k=1) + np.diag(np.ones(99), k=-1)
    T2 = np.diag(-np.ones(99)/2, k=1) + np.diag(-np.ones(98), k=2) + np.diag(np.ones(99)/2, k=-1) + np.diag(np.ones(98), k=-2)
    
    Q1 = lambda t :  Id @ expm(t *T1)
    Q2 = lambda t :  Id @ expm(t *T2)
    
    return Q1(t) @ (A1 + np.exp(t) * A2) @ Q2(t).T

def A_dot(t):
    """
    Generate the time-derivative of A-matrix as described in task 4
    input: 
    t - time variable
    epsilon - weighting variable for A1 and A2
    output:
    the derivative of matrix A evaluated at time t
    """

    Id = np.identity(100)
    
    T1 = np.diag(-np.ones(99), k=1) + np.diag(np.ones(99), k=-1)
    T2 = np.diag(-np.ones(99)/2, k=1) + np.diag(-np.ones(98), k=2) + np.diag(np.ones(99)/2, k=-1) + np.diag(np.ones(98), k=-2)
    
    Q1 = lambda t :  Id @ expm(t *T1)
    Q2 = lambda t :  Id @ expm(t *T2)
    
    # using chain rule
    return T1 @ Q1(t)@ (A1 + np.exp(t)*A2) @ Q2(t).T + Q1(t) @ ( A1 + np.exp(t) * A2) @ (T2 @ Q2(t)).T + np.exp(t) * Q1(t) @ A2 @ Q2(t).T

def A_2(t):
    """
    Generate A-matrix as described in task 5
    input: 
    t - time variable
    epsilon - weighting variable for A1 and A2
    output:
    the matrix A evaluated at time t
    """
    
    Id = np.identity(100)
    
    T1 = np.diag(-np.ones(99), k=1) + np.diag(np.ones(99), k=-1)
    T2 = np.diag(-np.ones(99)/2, k=1) + np.diag(-np.ones(98), k=2) + np.diag(np.ones(99)/2, k=-1) + np.diag(np.ones(98), k=-2)
    
    Q1 = lambda t :  Id @ expm(t *T1)
    Q2 = lambda t :  Id @ expm(t *T2)
    
    return Q1(t) @ (A1 + np.cos(t) * A2) @ Q2(t).T


def A_2_dot(t):
    """
    Generate the time-derivative of A-matrix as described in task 4
    input: 
    t - time variable
    epsilon - weighting variable for A1 and A2
    output:
    the derivative of matrix A evaluated at time t
    """

    Id = np.identity(100)
    
    T1 = np.diag(-np.ones(99), k=1) + np.diag(np.ones(99), k=-1)
    T2 = np.diag(-np.ones(99)/2, k=1) + np.diag(-np.ones(98), k=2) + np.diag(np.ones(99)/2, k=-1) + np.diag(np.ones(98), k=-2)
    
    Q1 = lambda t :  Id @ expm(t *T1)
    Q2 = lambda t :  Id @ expm(t *T2)
    
    # using chain rule
    return  T1 @ Q1(t)@ (A1 + np.cos(t)*A2) @ Q2(t).T + Q1(t) @ ( A1 + np.cos(t) * A2) @ (T2@Q2(t)).T - np.sin(t) * Q1(t) @ A2 @ Q2(t).T


np.random.seed(1)
A1_ep_103 = generateA(10**-3)
A2_ep_103 = generateA(10**-3)

def A_3(t):
    """
    Generate A-matrix as described in task 4
    input: 
    t - time variable
    epsilon - weighting variable for A1 and A2
    output:
    the matrix A evaluated at time t
    """
    
    Id = np.identity(100)
    
    T1 = np.diag(-np.ones(99), k=1) + np.diag(np.ones(99), k=-1)
    T2 = np.diag(-np.ones(99)/2, k=1) + np.diag(-np.ones(98), k=2) + np.diag(np.ones(99)/2, k=-1) + np.diag(np.ones(98), k=-2)
    
    Q1 = lambda t :  Id @ expm(t *T1)
    Q2 = lambda t :  Id @ expm(t *T2)
    
    return Q1(t) @ (A1_ep_103 + np.exp(t) * A2_ep_103) @ Q2(t).T


def A_3_dot(t):
    """
    Generate the time-derivative of A-matrix as described in task 4
    input: 
    t - time variable
    epsilon - weighting variable for A1 and A2
    output:
    the derivative of matrix A evaluated at time t
    """

    Id = np.identity(100)
    
    T1 = np.diag(-np.ones(99), k=1) + np.diag(np.ones(99), k=-1)
    T2 = np.diag(-np.ones(99)/2, k=1) + np.diag(-np.ones(98), k=2) + np.diag(np.ones(99)/2, k=-1) + np.diag(np.ones(98), k=-2)
    
    Q1 = lambda t :  Id @ expm(t *T1)
    Q2 = lambda t :  Id @ expm(t *T2)
    
    # using chain rule
    return (T1 @ Q1(t) @ A1_ep_103 + np.exp(t)*(Id + T1)@ Q1(t) @ A2_ep_103) @ Q2(t).T + (Q1(t) @ A1_ep_103 + np.exp(t) * Q1(t) @ A2_ep_103) @ (T2@Q2(t)).T
