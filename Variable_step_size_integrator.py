import numpy as np
import cay_operator as cay
import Example_matrices as ex

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
            if sigma > (1/16)*tol:
                R = 1
            else:
                R = 2

        t_new = t
        h_new = R*h
    return t_new,h_new

def construct_U_S_V_0_k(k,A):
    U,S,VT = np.linalg.svd(A) #  svd returns V.T and not V

    Sigma = np.diag(S[:k])
    U = U[:,:k]
    VT = VT[:k,:]

    return U,Sigma,VT.T # must return V.T (v.T.T = V) for rest of code to work as intended

#change cay_operator to cay_operator_facotrized
def variable_solver(t0,tf,A,tol,h0,method,k):
    U, S, V = construct_U_S_V_0_k(k,A) # construct initial conditions
    
    U_tensor = U
    S_tensor = S
    V_tensor = V
    t = t0
    h = h0
    j = 0
    count = 0

    t_vals = [0]
    while t < tf:
        K1_U,K1_V,S05,K1_S,U1,S1,V1 = method(h,t,U,V,S)

        S1_est = S05 + 0.5*K1_S
        U1_est = cay.cay_factorized(K1_U)@U
        V1_est = cay.cay_factorized(K1_V)@V

        sigma = np.linalg.norm(U1@S1@V1.T - U1_est@S1_est@V1_est.T,'fro')
        t = t + h
        t_new,h_new = step_control(sigma,tol,h,t)

        if t_new < t and count <= 3: # reject and try again with new 
            t = t_new
            h = h_new
            count += 1
        else: # accept 
            U,S,V = U1,S1,V1
            h_old = h
            h = h_new
            j += 1
            count = 0
            U_tensor = np.hstack((U_tensor,U))
            S_tensor = np.hstack((S_tensor,S))
            V_tensor = np.hstack((V_tensor,V))
            t_vals.append(t)

    if t > tf: # recomputing last step
        t = t-h_old
        h = tf-t
        _,_,_,_,U1,S1,V1 = method(h,t,U,V,S)
        U_tensor = np.hstack((U_tensor,U1))
        S_tensor = np.hstack((S_tensor,S1))
        V_tensor = np.hstack((V_tensor,V1))
        j += 1
        t_vals[-1] = tf

    return U_tensor,S_tensor,V_tensor,t_vals


def format_Yt(A,U,S,V,t_vals):
    """
    Converts the concatenated Y-matrix from a wide matrix to a 3D array
    """
    m,n = A.shape
    k = np.shape(S)[0]
    len_t = int(S.shape[1]/k)
    Ut = np.zeros((len_t,m,k))
    St = np.zeros((len_t,k,k))
    Vt = np.zeros((len_t,n,k)) # Not transposed here
    Yt = np.zeros((len_t,m,n))

    #sliceing up the matrices
    for i in range(len_t):
        Ut[i,:,:] = U[:,i*k:(i+1)*k]
        St[i,:,:] = S[:,i*k:(i+1)*k]
        Vt[i,:,:] = V[:,i*k:(i+1)*k]
        Yt[i] = Ut[i]@St[i]@Vt[i].T

    #recomputing last step
    Ut[-2,:,:] = Ut[-1,:,:]
    St[-2,:,:] = St[-1,:,:]
    Vt[-2,:,:] = Vt[-1,:,:] 
    Yt[-2,:,:] = Yt[-1,:,:]

    return Yt[:-1],Ut[:-1],St[:-1],Vt[:-1],t_vals


# can be removed later
def format_SVD(A,X):
    """
    Converts the concatenated X-matrix from a wide matrix to a 3D array
    """
    m,n = A.shape
    len_t = int(X.shape[1]/n)
    Xt = np.zeros((len_t,m,n))

    for i in range(len_t):
        Xt[i,:,:] = X[:,i*m:(i+1)*m]

    return Xt


def extract_singular_values(S_matrix):
    len_t,k = np.shape(S_matrix)[:2]
    S = np.zeros((len_t,k))
    for i in range(len_t):   
       S[i,:] = np.diag(S_matrix[i])
    return S.T

def compute_singular_values(A,k,t_vals):
    sing_vals = np.linalg.svd(A(0))[1][:k]
    for t in t_vals[1:]:
        s = np.linalg.svd(A(t))[1][:k]
        sing_vals = np.vstack((sing_vals,s))
    return sing_vals.T