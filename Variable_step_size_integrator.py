import numpy as np
import dynamic_low_rank as dlr 

def step_control(sigma,tol,h,t):
    if sigma > tol:
        print('action 1','sigma',sigma,'tol',tol)
        t_new = t-h
        h_new = 0.5*h
    else:
        if sigma > 0.5*tol:
            print('action 2','sigma',sigma,'tol',tol)
            R = (tol/sigma)**(1/3)
            if R > 0.9 or R < 0.5: 
                R = 0.7
        else:
            if sigma > (1/16)*tol:
                print('action 3','sigma',sigma,'tol',1/16*tol)
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


def variable_solver(t0,tf,A_dot,tol,h0,method):
    U, S, V = construct_U_S_V_0(A_dot) # construct initial conditions
    t = t0
    h = h0
    j = 0
    count = 0
    while t < tf:
        q = np.linalg.norm
        print('count',count,'j',j,'t',t,'h',h,'u',q(U),'v',q(V),'s',q(S))
        K1_U,K1_V,S05,K1_S,U1,S1,V1 = method(h,t,U,V,S)
        
        S1_est = S05 + 0.5*K1_S
        U1_est = dlr.cay_operator(K1_U)@U
        V1_est = dlr.cay_operator(K1_V)@V

        sigma = np.linalg.norm(U1@S1@V1.T-U1_est@S1_est@V1_est.T,'fro')
        
        t = t + h
        t_new,h_new = step_control(sigma,tol,h,t)

        if t_new < t and count <= 3: # reject and try again
            K1_U,K1_V,S05,K1_S,U1,S1,V1 = method(h_new,t_new,U,V,S)
        
            t = t_new
            h = h_new
            count += 1
        else: # accept 
            # missing update of U,S,V
            h_old = h
            h = h_new
            j += 1
            count = 0

    if t > tf: # recomputing last step
        t = t-h_old
        h = tf-t
        _,_,_,_,U1,S1,V1 = method(h,t,U,V,S)
        j += 1

    return U1,S1,V1,j