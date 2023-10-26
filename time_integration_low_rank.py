import numpy as np
import dynamic_low_rank as dlr
## gjør mer grunndig

def second_order_method(h,U,A_dot,V,S,tol):
    m,n = U.shape[0],V.shape[0]
    I_mm = np.eye(m)
    K1_S = h*U.T@A_dot@V
    S05 = S + 0.5*K1_S

    FUj = dlr.FU(U,A_dot,V,S)
    K1_U = h*(FUj@U.T-U@FUj.T)
    U05 = dlr.cay_operator(0.5*K1_U)@U

    FVj = dlr.FV(V,A_dot,U,S)
    K1_V = h*(FVj@V.T-V@FVj.T)
    V05 = dlr.cay_operator(0.5*K1_V)@V

    # her skal A_dot endres til A_dot05
    K2_S = h*U05.T@A_dot@V
    S1 = S + 0.5*K2_S

    FU05 = dlr.FU(U05,A_dot,V05,S05) 
    K2_U = h*(FU05@U05.T-U05@FU05.T)
    U1 = dlr.cay_operator(K2_U)@U

    FV05 = dlr.FV(V05,A_dot,U05,S05)
    K2_V = h*(FV05@V05.T-V05@FV05.T)
    V1 = dlr.cay_operator(K2_V)@V

    return K1_U,K1_V,S05,K1_S,U1,S1,V1
    