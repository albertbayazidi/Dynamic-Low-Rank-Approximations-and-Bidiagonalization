import numpy as np

#Naive method
def cay_operator(B):
    I = np.eye(B.shape[0])
    inv = np.linalg.inv((I-0.5*B)) 
    return inv@(I+0.5*B)

#QR method
def cay_operator_QR(F,U):
    C = np.array(F,-U)
    print(C)

A = np.array([[1,2,3],[4,5,6],[7,8,9]])
B = np.array([[1,2,3],[4,5,6],[7,8,9]])

cay_operator_QR(A,B)

def FU(U,A_dot,V,S):
    I_mm = np.eye(U.shape[0])
    return (I_mm-U@U.T)@A_dot@V@np.linalg.inv(S)

def FV(V,A_dot,U,S):
    I_mm = np.eye(U.shape[0])
    return (I_mm-V@V.T)@A_dot.T@U@np.linalg.inv(S).T

def second_order_method(h,U,A_dot,V,S,tol):
    I_mm = np.eye(U.shape[0])
    K1_S = h*U.T@A_dot@V
    S05 = S + 0.5*K1_S

    FUj = FU(U,A_dot,V,S)
    K1_U = h*(FUj@U.T-U@FUj.T)
    U05 = cay_operator(0.5*K1_U)@U

    FVj = FV(V,A_dot,U,S)
    K1_V = h*(FVj@V.T-V@FVj.T)
    V05 = cay_operator(0.5*K1_V)@V

    # her skal A_dot endres til A_dot05
    K2_S = h*U05.T@A_dot@V
    S1 = S + 0.5*K2_S

    FU05 = FU(U05,A_dot,V05,S05) 
    K2_U = h*(FU05@U05.T-U05@FU05.T)
    U1 = cay_operator(K2_U)@U

    FV05 = FV(V05,A_dot,U05,S05)
    K2_V = h*(FV05@V05.T-V05@FV05.T)
    V1 = cay_operator(K2_V)@V

    S1_est = S05 + 0.5*K1_S
    U1_est = cay_operator(K1_U)@U
    V1_est = cay_operator(K1_V)@V

    sigma = np.linalg.norm(U1@S1@V1.T-U1_est@S1_est@V1_est.T,'fro')

    h = h*(sigma/tol)**(-1/3)