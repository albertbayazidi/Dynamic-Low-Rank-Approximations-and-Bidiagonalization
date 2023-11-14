import numpy as np
import Example_matrices as ex
import cay_operator as cay

def FU(U,A_dot,V,S):
    I_mm = np.eye(U.shape[0])
    S_inv = np.linalg.inv(S)
    return (I_mm-U@U.T)@A_dot@V@S_inv 

def FV(V,A_dot,U,S):
    I_mm = np.eye(V.shape[0])
    S_inv = np.linalg.inv(S)
    return (I_mm-V@V.T)@A_dot.T@U@S_inv.T 


def second_order_method(h,t,U,V,S): 
    """
    set of ODES to be solved
    input
    h: step size
    t: current time
    U,V,S: low rank matrices
    output
    K1_U,K1_V,S05,K1_S,U1,S1,V1: the different steps in the RK2 method
    """
    
    A_dot = ex.A_dot(t) 
    K1_S = h*U.T@A_dot@V
    S05 = S + 0.5*K1_S

    FUj = FU(U,A_dot,V,S)
    K1_U = np.array([h*FUj,h*U])
    U05 = cay.cay_factorized(0.5*K1_U)@U 

    FVj = FV(V,A_dot,U,S)
    K1_V = np.array([h*FVj,h*V])
    V05 = cay.cay_factorized(0.5*K1_V)@V 

    A_dot05 = ex.A_dot(t+h/2)
    K2_S = h*U05.T@A_dot05@V05
    S1 = S + K2_S

    FU05 = FU(U05,A_dot05,V05,S05) 
    K2_U = np.array([h*FU05,h*U05])
    U1 = cay.cay_factorized(K2_U)@U 

    FV05 = FV(V05,A_dot05,U05,S05)
    K2_V = np.array([h*FV05,h*V05])
    V1 = cay.cay_factorized(K2_V)@V 

    return K1_U,K1_V,S05,K1_S,U1,S1,V1
    

def second_order_method2(h,t,U,V,S): 
    """
    set of ODES to be solved
    input
    h: step size
    t: current time
    U,V,S: low rank matrices
    output
    K1_U,K1_V,S05,K1_S,U1,S1,V1: the different steps in the RK2 method
    """

    A_dot = ex.A_2_dot(t) 
    K1_S = h*U.T@A_dot@V
    S05 = S + 0.5*K1_S

    FUj = FU(U,A_dot,V,S)
    K1_U = np.array([h*FUj,h*U])
    U05 = cay.cay_factorized(0.5*K1_U)@U 

    FVj = FV(V,A_dot,U,S)
    K1_V = np.array([h*FVj,h*V])
    V05 = cay.cay_factorized(0.5*K1_V)@V 

    A_dot05 = ex.A_2_dot(t+h/2)
    K2_S = h*U05.T@A_dot05@V05
    S1 = S + K2_S

    FU05 = FU(U05,A_dot05,V05,S05) 
    K2_U = np.array([h*FU05,h*U05])
    U1 = cay.cay_factorized(K2_U)@U 

    FV05 = FV(V05,A_dot05,U05,S05)
    K2_V = np.array([h*FV05,h*V05]) 
    V1 = cay.cay_factorized(K2_V)@V 

    return K1_U,K1_V,S05,K1_S,U1,S1,V1
    