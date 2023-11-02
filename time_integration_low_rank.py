import numpy as np
import dynamic_low_rank as dlr

def A_fun(t,n,m):
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,m)
    X,Y = np.meshgrid(x,y)
    A = np.exp(-t)*np.sin(np.pi*X)*np.sin(2*np.pi*Y) #removed 5pi^2
    return A

def laplace(n,m):
    ones = np.ones((n)-2)
    k = 1/n 
    L =  1/k**2*(2*np.diag(ones) - np.diag(ones[:-1],-1) - np.diag(ones[:-1],1)) 
    return np.pad(L,1)

def A_dot_fun(t,n,m):
    A = A_fun(t,n,m)
    L = laplace(n,m)
    A_dot = L@A + A@L
    return A_dot.reshape(n,m)


def FU(U,Q,m):
    I = np.eye(m)
    return (I-U@U.T)@Q@U

def FV(V,R,n):
    I = np.eye(n)
    return (I-V@V.T)@R@V

#Time integration of a low-rank linear matrix ODE
def second_order_method(h,t,U,V,S): # takes in t as a dummyvariable so it works with the same solver
    L = laplace(U.shape[0],V.shape[0]) 
    Q = L
    R = L
    m,n = U.shape[0],V.shape[0]

    K1_S = h*(U.T@Q@U@S + S@V.T@R@V)
    S05 = S + 0.5*K1_S

    FUj = FU(U,Q,m)
    K1_U = h*(FUj@U.T-U@FUj.T)
    U05 = dlr.cay_operator(0.5*K1_U)@U

    FVj = FV(V,R,n)
    K1_V = h*(FVj@V.T-V@FVj.T)
    V05 = dlr.cay_operator(0.5*K1_V)@V

    K2_S = h*(U05.T@Q@U05@S05 + S05@V05.T@R@V05) # blir ikke brukt i denne metoden
    S1 = S + h*(U.T@Q@U@S + S@V.T@R@V)

    FU05 = FU(U05,Q,m) 
    K2_U = h*(FU05@U05.T-U05@FU05.T)
    U1 = dlr.cay_operator(K2_U)@U

    FV05 = FV(V05,R,n)
    K2_V = h*(FV05@V05.T-V05@FV05.T)
    V1 = dlr.cay_operator(K2_V)@V

    return K1_U,K1_V,S05,K1_S,U1,S1,V1
    