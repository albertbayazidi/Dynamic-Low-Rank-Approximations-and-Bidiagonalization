import numpy as np

def A_fun(t,n,m):
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,m)
    X,Y = np.meshgrid(x,y)
    A = np.exp(-t)*np.sin(np.pi*X)*np.sin(2*np.pi*Y) #removed 5pi^2
    return A

def laplace(m,n):
    ones = np.ones(m*n)
    k = np.min([1/n,1/m]) # dobbelsjekk
    L =  1/k**2*(2*np.diag(ones) - np.diag(ones[:-1],-1) - np.diag(ones[:-1],1)) 
    return L


def A_dot_fun(t,n,m):
    A = A_fun(t,n,m)
    L = laplace(n,m)
    A_dot = L@A.ravel() + A.ravel()@L
    return A_dot.reshape(n,m)


#Naive method
def cay_operator(B):
    I = np.eye(B.shape[0])
    inv = np.linalg.inv((I-0.5*B)) 
    return inv@(I+0.5*B)

#First method
#https://www.sciencedirect.com/science/article/pii/S0898122101002784

#QR method not finished
def cay_operator_QR(F,U):
    C = np.block([F,-U])
    D = np.block([U,F])

def FU(U,A_dot,V,S):
    I_mm = np.eye(U.shape[0])
    return (I_mm-U@U.T)@A_dot@V@np.linalg.inv(S)

def FV(V,A_dot,U,S):
    I_mm = np.eye(U.shape[0])
    return (I_mm-V@V.T)@A_dot.T@U@np.linalg.inv(S).T

def second_order_method(h,t,U,V,S):
    A_dot = A_dot_fun(t,n,m)
    m,n = U.shape[0],V.shape[0]
    I_mm = np.eye(m)
    K1_S = h*U.T@A_dot@V
    S05 = S + 0.5*K1_S

    FUj = FU(U,A_dot,V,S)
    K1_U = h*(FUj@U.T-U@FUj.T)
    U05 = cay_operator(0.5*K1_U)@U

    FVj = FV(V,A_dot,U,S)
    K1_V = h*(FVj@V.T-V@FVj.T)
    V05 = cay_operator(0.5*K1_V)@V

    # her skal A_dot endres til A_dot05
    A_dot05 = A_dot_fun(t+h/2,n,m)
    K2_S = h*U05.T@A_dot05@V
    S1 = S + 0.5*K2_S

    FU05 = FU(U05,A_dot05,V05,S05) 
    K2_U = h*(FU05@U05.T-U05@FU05.T)
    U1 = cay_operator(K2_U)@U

    FV05 = FV(V05,A_dot05,U05,S05)
    K2_V = h*(FV05@V05.T-V05@FV05.T)
    V1 = cay_operator(K2_V)@V

    return K1_U,K1_V,S05,K1_S,U1,S1,V1
    