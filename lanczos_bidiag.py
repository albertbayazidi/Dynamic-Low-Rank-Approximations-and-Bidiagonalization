import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags

def lanczosBiDiag(A,k,b, orthogonalize = False):
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
    if (orthogonalize):
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
    else:
        for i in range(1,k):
            beta[i-1] = np.linalg.norm(A@v_i - alpha[i-1] * u_i)
            u_i = (A@v_i - alpha[i-1] * u_i)/beta[i-1]
            Pk[:,i] = u_i

            alpha[i] = np.linalg.norm(A.T@u_i - beta[i-1] * v_i)
            v_i = (A.T@u_i - beta[i-1] * v_i)/alpha[i]
            Qk[:,i] = v_i  
    
    Bk = diags([alpha,beta],[0,-1]).toarray()
    return Pk,Qk,Bk


def plotSingularValues(A):
    """
    Compute the singular values of matrix A, and plot them
    input: 
    A -  matrix for which is to be done SVD on.
    """
    sing_vals = np.linalg.svd(A)[1]
    plt.plot([i for i in range(1,len(sing_vals)+1)],sing_vals)
    plt.grid(True)
    plt.title("Singular values")
    plt.xlabel('singular value number')
    plt.ylabel('singular value')
    plt.show()

def compareApproximations(A):
    """
    Compares the Frobenius norm of resulting matrices when using Lanczos bidiagonalization and SVD decomposition.
    This is done for all values of k from 1 to n. Result is then plotted.
    input: 
    A - matrix to be approximated
    """
    m,n = np.shape(A)
    b = np.ones(m)
    P,Q,B = lanczosBiDiag(A,n,b, orthogonalize = True)
    U,S,VT = np.linalg.svd(A)
    
    svd_norm = np.zeros(n)
    bidiag_norm = np.zeros(n)
    for k in range(1,n):
        
        Pk = P[:,:k]
        Qk = Q[:,:k]
        Bk = B[:k,:k]
        A_bidiag = Pk @ Bk @ Qk.T
        
        Uk = U[:, :k]  
        Sk = np.diag(S[:k])  
        VTk = VT[:k, :]
        
        A_svd = np.dot(Uk, np.dot(Sk, VTk))
        svd_norm[k-1] = np.linalg.norm(A-A_svd)
        bidiag_norm[k-1] = np.linalg.norm(A-A_bidiag)
        
    plt.title("Frobenius norm of SVD and Lanczos bidiagonalization matrices")
    plt.ylabel("Frobenius normed error")
    plt.xlabel("value of k")
    plt.plot([i for i in range(1,n+1)],svd_norm, label = "SVD")
    plt.plot([i for i in range(1,n+1)],bidiag_norm, label = "Lanczos")
    plt.legend()
    plt.show()