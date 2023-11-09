import numpy as np
import cay_operator as cay

# change this function to comunicate with how u should be definde
# should be square matrix
def u_fun(g,n,m):
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,m)
    X,Y = np.meshgrid(x,y)
    A = g(X,Y)
    
    A[:,0] = 0
    A[0,:] = 0
    A[:,-1] = 0
    A[-1,:-1] = 0

    return A

# should be square matrix
def laplace(n): 
    # n = dim of L - matrix (which is for N+1 points)
    ones = np.ones((n)-2)
    k = 1/(n-1) 
    L =  1/k**2*(-2*np.diag(ones) + np.diag(ones[:-1],-1) + np.diag(ones[:-1],1)) 
    return np.pad(L,1)

def u_dot_fun(g,n,m):
    u = u_fun(g,n,m)
    L = laplace(n)
    A_dot = L@u + u@L
    return A_dot.reshape(n,m)

def FU(U,Q,m):
    I = np.eye(m)
    return (I-U@U.T)@Q@U

def FV(V,R,n):
    I = np.eye(n)
    return (I-V@V.T)@R.T@V

#Time integration of a low-rank linear matrix ODE
def second_order_method(h,t,U,V,S): # takes in t as a dummyvariable so it works with the same solver
    L = laplace(U.shape[0]) 
    Q = L
    R = L
    m,n = U.shape[0],V.shape[0]

    K1_S = h*(U.T@Q@U@S + S@V.T@R@V) 
    S05 = S + 0.5*K1_S 

    FUj = FU(U,Q,m) 
    #K1_U = h*(FUj@U.T-U@FUj.T) 
    K1_U = [h*0.5*FUj,h*0.5*U]
    U05 = cay.cay_factorized(K1_U)@U # droped 0.5

    FVj = FV(V,R,n) 
    #K1_V = h*(FVj@V.T-V@FVj.T)
    K1_V = [h*0.5*FVj,h*0.5*V]
    V05 = cay.cay_factorized(K1_V)@V # droped 0.5

    K2_S = h*(U05.T@Q@U05@S05 + S05@V05.T@R@V05)  
    #S1 = S + h*(U.T@Q@U@S + S@V.T@R@V)
    S1 = S + K2_S

    FU05 = FU(U05,Q,m) 
    #K2_U = h*(FU05@U05.T-U05@FU05.T)
    K2_U = [h*FU05,h*U05]
    U1 = cay.cay_factorized(K2_U)@U  # droped 0.5

    FV05 = FV(V05,R,n) 
    #K2_V = h*(FV05@V05.T-V05@FV05.T)
    K2_V = [h*FV05,h*V05]
    V1 = cay.cay_factorized(K2_V)@V # droped 0.5

    return K1_U,K1_V,S05,K1_S,U1,S1,V1
    