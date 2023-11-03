import numpy as np
import Example_matrices as ex

#Naive method
def cay_operator(B):
    I = np.eye(B.shape[0])
    inv = np.linalg.inv((I-0.5*B)) 
    return inv@(I+0.5*B)

def cay_factorized(F,mat):
    C = np.block([F,-mat])
    D = np.block([mat,F])

    I_int = np.eye(mat.shape[1])
    O = np.zeros((F.shape[1],F.shape[1]))

    DTC = np.block([[O,-I_int]
                    ,[F.T@F,O]])
    
    I_eq = np.eye(DTC.shape[0])
    DTC_inv = np.linalg.inv(I_eq-0.5*DTC)
    I_fin = np.eye(F.shape[0])

    return I_fin + C@DTC_inv@D.T

def cay_factorized_optim(F,mat):
    C = np.block([F,-mat])
    D = np.block([mat,F])
    I_int = np.eye(mat.shape[1])

    FTF = F.T@F    
    A = np.linalg.inv(I_int+0.25*FTF)
    temp = A@FTF
    B = -0.5*A
    C_int = 0.5*temp
    D_int = I_int - 0.25*temp

    DTC_inv = np.block([[A,B],
                        [C_int,D_int]])

    I_fin = np.eye(F.shape[0])
    return I_fin + C@DTC_inv@D.T


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
    K1_U = h*(FUj@U.T-U@FUj.T)
    U05 = cay_operator(0.5*K1_U)@U

    FVj = FV(V,A_dot,U,S)
    K1_V = h*(FVj@V.T-V@FVj.T)
    V05 = cay_operator(0.5*K1_V)@V

    A_dot05 = ex.A_dot(t+h/2)
    K2_S = h*U05.T@A_dot05@V05
    S1 = S + K2_S

    FU05 = FU(U05,A_dot05,V05,S05) 
    K2_U = h*(FU05@U05.T-U05@FU05.T)
    U1 = cay_operator(K2_U)@U

    FV05 = FV(V05,A_dot05,U05,S05)
    K2_V = h*(FV05@V05.T-V05@FV05.T)
    V1 = cay_operator(K2_V)@V

    return K1_U,K1_V,S05,K1_S,U1,S1,V1
    