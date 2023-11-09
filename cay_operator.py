import numpy as np

#Naive method
def cay_operator(Obj):
    # Obj = [F,mat]
    F = Obj[0]
    mat = Obj[1]
    B = F@mat.T-mat@F.T 
    I = np.eye(B.shape[0])
    inv = np.linalg.inv((I-0.5*B)) 
    return inv@(I+0.5*B)

def cay_factorized(Obj):
    # Obj = [F,mat]
    F = Obj[0]
    mat = Obj[1]
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

def cay_factorized_optim(Obj):
    F = Obj[0]
    mat = Obj[1]
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