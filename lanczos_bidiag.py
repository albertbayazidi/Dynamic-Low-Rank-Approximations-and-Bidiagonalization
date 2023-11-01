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


def plotSingularValues(A1,A2,A3):
    """
    Compute the singular values of matrix A, and plot them
    input: 
    A_i - matrices for which is to be done SVD on.
    """
    fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize = (20,5))

    sing_vals_A1 = np.linalg.svd(A1)[1]
    sing_vals_A2 = np.linalg.svd(A2)[1]  
    sing_vals_A3 = np.linalg.svd(A3)[1]
    
    ax1.plot([i for i in range(1,len(sing_vals_A1)+1)],sing_vals_A1)
    ax2.plot([i for i in range(1,len(sing_vals_A2)+1)],sing_vals_A2)
    ax3.plot([i for i in range(1,len(sing_vals_A3)+1)],sing_vals_A3)
    
    
    ax1.set_title("Singular values for A1")
    ax2.set_title("Singular values for A2")
    ax3.set_title("Singular values for A3")
    
    for ax in (ax1, ax2,ax3):
        ax.set(xlabel='singular value number', ylabel='singular value')
        ax.grid(True)


def compareApproximations(A,b):
    """
    Compares the Frobenius norm of resulting matrices when using Lanczos bidiagonalization and SVD decomposition.
    This is done for all values of k from 1 to n. Result is then plotted.
    input: 
    A - matrix to be approximated
    """
    m,n = np.shape(A)
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
    
    plt.figure(figsize = (10,6))
    plt.title("Frobenius norm of SVD and Lanczos bidiagonalization matrices")
    plt.grid(True)
    plt.ylabel("Frobenius normed error")
    plt.xlabel("value of k")
    plt.plot([i for i in range(1,n+1)],svd_norm, label = "SVD")
    plt.plot([i for i in range(1,n+1)],bidiag_norm,"--", label = "Lanczos")
    plt.legend()
    plt.show()

def orthogonalityError(A,b):
    """
    calculates the matrices resulting from Lanczos bidiagonalization, 
    with and without reorthogonalization. Then calculates orthogonality error,
    measured as the mean dot product between vectors in P and Q respectively.
    input: 
    A - matrix that is to be approximated by Lanczos 
    b - rhs of original linear system of equations
    """
    m,n = np.shape(A)
    P,Q = lanczosBiDiag(A,n,b, orthogonalize = False)[:-1]
    P_ortho,Q_ortho = lanczosBiDiag(A,n,b, orthogonalize = True)[:-1]
    
    # Find mean sum of unique dot-products between columns in P and Q, e.g.
    # err_P = sqrt(sum(v_i dot v_j)) for all i={1,..,n} and j = {1,..,n} where j != i
    # this is equivalent to calculating P.T @ P, extracting the upper triangular matrix
    # (without the diagonal), and taking the Frobenius norm times the dimension of P.
    
    err_P = 1/n * np.linalg.norm(np.triu(np.dot(P.T,P),1))
    err_Q = 1/n * np.linalg.norm(np.triu(np.dot(Q.T,Q),1))
    
    err_P_ortho =  1/n * np.linalg.norm(np.triu(np.dot(P_ortho.T,P_ortho),1))
    err_Q_ortho = 1/n *  np.linalg.norm(np.triu(np.dot(Q_ortho.T,Q_ortho),1))
    
    print("Without re-orthogonalization: ")
    print("Mean error for P: ", "{:.3e}".format(err_P), "  Mean error for Q: ", "{:.3e}".format(err_Q))
    print("With re-orthogonalization: ")
    print("Mean error for P: ", "{:.3e}".format(err_P_ortho)," Mean error for Q: ", "{:.3e}".format(err_Q_ortho))
    