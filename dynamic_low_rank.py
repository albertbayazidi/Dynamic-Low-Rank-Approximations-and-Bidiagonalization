import numpy as np
import Example_matrices as ex
import cay_operator as cay

def FU(U,A_dot,V,S):
    I_mm = np.eye(U.shape[0])
    S_inv = np.linalg.inv(S)
    return (I_mm-U@U.T)@A_dot@V@S_inv # can change inverse of S to 1/S

def FV(V,A_dot,U,S):
    I_mm = np.eye(V.shape[0])
    S_inv = np.linalg.inv(S)
    return (I_mm-V@V.T)@A_dot.T@U@S_inv.T # can change inverse of S to 1/S

#most change cay operator to cay factorized
def second_order_method(h,t,U,V,S): 
    A_dot = ex.A_dot(t) 
    K1_S = h*U.T@A_dot@V
    S05 = S + 0.5*K1_S

    FUj = FU(U,A_dot,V,S)
    #K1_U = h*(FUj@U.T-U@FUj.T)
    K1_U = [h*0.5*FUj,h*0.5*U]
    U05 = cay.cay_factorized(K1_U)@U # droped 0.5

    FVj = FV(V,A_dot,U,S)
    #K1_V = h*(FVj@V.T-V@FVj.T)
    K1_V = [h*0.5*FVj,h*0.5*V]
    V05 = cay.cay_factorized(K1_V)@V # droped 0.5

    A_dot05 = ex.A_dot(t+h/2)
    K2_S = h*U05.T@A_dot05@V05
    S1 = S + K2_S

    FU05 = FU(U05,A_dot05,V05,S05) 
    #K2_U = h*(FU05@U05.T-U05@FU05.T)
    K2_U = [h*FU05,h*U05]
    U1 = cay.cay_factorized(K2_U)@U # droped 0.5

    FV05 = FV(V05,A_dot05,U05,S05)
    #K2_V = h*(FV05@V05.T-V05@FV05.T)
    K2_V = [h*FV05,h*V05]
    V1 = cay.cay_factorized(K2_V)@V # droped 0.5

    return K1_U,K1_V,S05,K1_S,U1,S1,V1
    