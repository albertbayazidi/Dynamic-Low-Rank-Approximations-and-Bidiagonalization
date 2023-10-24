import numpy as np
from scipy.sparse import diags

def lanczosBiDiag(A,k,b):
    """
    Performs Lanczos bidiagonalization algorithm
    input:
    A - matrix of dimension m x n
    k - rank of X(t)-matrix, also # largest singular values
    b - rhs vector of dimension m
    output:
    
    """
    m,n = np.shape(A)
    
    Pk = np.zeros((m,k))
    Qk = np.zeros((n,k))
    
    alpha = np.zeros(k)
    beta = np.zeros(k-1)
    
    beta_1 = np.linalg.norm(b)
    u_i = b / beta_1
    Atu = A.T @ u_i
    alpha[0] = np.linalg.norm(Atu)
    v_i = Atu / alpha[0]

    Pk[:,0] = u_i
    Qk[:,0] = v_i
    
    for i in range(1,k):
        beta[i-1] = np.linalg.norm(A@v_i - alpha[i-1] * u_i)
        u_i = (A@v_i - alpha[i-1] * u_i)/beta[i-1]
        Pk[:,i] = u_i
        
        alpha[i] = np.linalg.norm(A.T@u_i - beta[i-1] * v_i)
        v_i = (A.T@u_i - beta[i-1] * v_i)/alpha[i]
        Qk[:,i] = v_i  
    
    Bk = diags([alpha,beta],[0,-1]).toarray()
    return Pk,Qk,Bk
    

def lanczosBiDiagOrthogonalised(A,k,b):
    """
    Performs Lanczos bidiagonalization algorithm, with re-orthogonalization
    input:
    A - matrix of dimension m x n
    k - rank of X(t)-matrix, also # largest singular values
    b - rhs vector of dimension m
    output:
    
    """
    m,n = np.shape(A)
    
    Pk = np.zeros((m,k))
    Qk = np.zeros((n,k))
    
    alpha = np.zeros(k)
    beta = np.zeros(k-1)
    
    beta_1 = np.linalg.norm(b)
    u_i = b / beta_1
    Atu = A.T @ u_i
    alpha[0] = np.linalg.norm(Atu)
    v_i = Atu / alpha[0]

    Pk[:,0] = u_i
    Qk[:,0] = v_i
    
    for i in range(1,k):
        beta[i-1] = np.linalg.norm(A@v_i - alpha[i-1] * u_i)
        u_i = (A@v_i - alpha[i-1] * u_i)/beta[i-1]
        Pk[:,i] = u_i
        
        w_i = (A.T@u_i - beta[i-1] * v_i)
        
        for j in range(i):
            w_i -= np.dot(Qk[:,j-1],w_i) * Qk[:,j-1]
            
        alpha[i] = np.linalg.norm(w_i)
        v_i = w_i/alpha[i]
        Qk[:,i] = v_i  
    
    Bk = diags([alpha,beta],[0,-1]).toarray()
    return Pk,Qk,Bk