import numpy as np
import dynamic_low_rank as dlr 

def step_control(sigma,tol,h,t):
    if sigma > tol:
        t_new = t-h
        h_new = 0.5*h
    else:
        if sigma > 0.5*tol:
            R = (tol/sigma)**(1/3)
            if R > 0.9 or R < 0.5: 
                R = 0.7
        else:
            if sigma > 1/16*tol:
                R = 1
            else:
                R = 2
        t_new = t
        h_new = R*h
    return t_new,h_new


def construct_U_S_V_0(A):
    U,S,V = np.linalg.svd(A) # can use hermitain = True if symetric

    Sigma = np.zeros(A.shape)
    for i,s in enumerate(S):
        Sigma[i,i] = s
    
    return U,Sigma,V.T  #dobbelt sjekk


def variable_solver(t0,tf,A,tol,h0):
    U, S, V = construct_U_S_V_0(A)
    count = 0
    t = t0
    h = h0
    j = 0
    while t < tf:
        K1_U,K1_V,S05,K1_S,U1,S1,V1 = dlr.second_order_method(h,t,U,V,S)
        
        S1_est = S05 + 0.5*K1_S
        U1_est = dlr.cay_operator(K1_U)@U
        V1_est = dlr.cay_operator(K1_V)@V

        sigma = np.linalg.norm(U1@S1@V1.T-U1_est@S1_est@V1_est.T,'fro')
        t_old,h_old = t, h
        t_new,h_new = step_control(sigma,tol,h,t)

        if t_new < t and count <= 3:
            K1_U,K1_V,S05,K1_S,U1,S1,V1 = dlr.second_order_method(h_new,t_new,U,V,S)
        
            t = t_new
            h = h_new
            count += 1

        else:
            t,h,U,V,S = t_old,h_old,U1,V1,S1
            j += 1
            count = 0
    if t > tf:
        t = t-h_old
        h = tf-t
        _,_,_,_,U1,S1,V1 = dlr.second_order_method(h,U,V,S,tol)
        j += 1