import numpy as np
import dynamic_low_rank as dlr
import Variable_step_size_integrator as vssi
import time_integration_low_rank as tilr
import matplotlib.pyplot as plt
import Example_matrices as ex
import lanczos_bidiag as lb

def g(x,y):
    return np.sin(np.pi*x)*np.sin(2*np.pi*y)

def u_exact(x,y,t):
    return np.exp(-5*np.pi**2*t)*np.sin(np.pi*x)*np.sin(2*np.pi*y)


def u_exact_t(N,t_vals):
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    X,Y = np.meshgrid(x,y)

    u_exact_tensor = np.zeros((len(t_vals),N,N))
    for i,t in enumerate(t_vals):
        u_exact_tensor[i] = u_exact(X,Y,t)

    return u_exact_tensor

def compute_error_norm(t_vals,Yt,u_ex_t):
    error = np.zeros(len(t_vals)-1)
    for i in range(len(t_vals)-1):
        error[i] = np.linalg.norm(Yt[i]-u_ex_t[i])
    return error


def solve_heat_eq(k,h0,tf,tol):
    t0 = 0
    N = 32

    m,n = N,N
    u0 = tilr.u_fun(g,m,n) 

    # specify what method to use
    method = tilr.second_order_method

    # do method
    U,S,V,t_vals = vssi.variable_solver(t0,tf,u0,tol,h0,method,k)

    Yt,Ut,St,Vt = vssi.format_Yt(u0,U,S,V)
    u_ex_t = u_exact_t(N,t_vals)
    error = compute_error_norm(t_vals,Yt,u_ex_t)
    return Yt,Ut,Vt,u_ex_t,error

def plot_heat_sol_and_error(k,Yt,u_ex_t,error):
    # comparing the solution to the exact solution
    plt.figure(figsize=(15,5))
    plt.suptitle(f'Rank {k} Solution of heat and error of')
    plt.subplot(1,3,1)
    plt.title('nummerical solution')
    plt.imshow(Yt[-1])
    plt.subplot(1,3,2)
    plt.title('exact solution')
    plt.imshow(u_ex_t[-1])
    plt.subplot(1,3,3)
    plt.title('Error')
    plt.xlabel('iteration')
    plt.plot(error)
    plt.show()

# computes truncated SVD oF A for all t in t_vals
def compute_SVD_t(A,k,t_vals):
    U,S,VT = np.linalg.svd(A(0))
    S = np.diag(S[:k])
    U = U[:,:k]
    VT = VT[:k,:]
    X0 = U@S@VT
    X = X0
    for t in t_vals:
        U,S,VT = np.linalg.svd(A(t))
        S = np.diag(S[:k])
        U = U[:,:k]
        VT = VT[:k,:]
        X_temp = U@S@VT
        X = np.hstack((X,X_temp))
    
    Xt = vssi.format_result(A(0),X)

    return Xt

# computes the rank k SVD apprixmation of A_dot for all t in t_vals
def construct_Y_dot(U,S,V,tvals):
    n = U[0,:,:].shape[0]
    Yt_dot = np.zeros((len(tvals),n,n))
    I = np.eye(n,n)
    for i,t in enumerate(tvals):
        A_dot = ex.A_dot(t)
        S_dot = U[i].T@A_dot@V[i]
        U_dot = (I-U[i]@U[i].T) @ A_dot @ V[i] @ np.linalg.inv(S[i])
        V_dot = (I-V[i]@V[i].T) @ A_dot @ U[i] @ np.linalg.inv(S[i].T)
        Yt_dot[i] = U_dot@S[i]@V[i].T + U[i]@S_dot@V[i].T + U[i]@S[i]@V_dot.T

    return Yt_dot

# computes the rank k SVD for A using the lanczos bidiagonalization method
def construct_Wt(A,k,tvals):
    # A should be a function
    n = A(0).shape[0]
    Wt = np.zeros((len(tvals),n,n))
    b = np.ones(n)
    for i,t in enumerate(tvals):
        Pk,Qk,Bk = lb.lanczosBiDiag(A(t),k,b, True)
        Wt[i,:,:] = Pk@Bk@Qk.T
    return Wt

def solve_task4(k,A,h0,tf,tol):
    # A should be a function
    t0 = 0
    A0 = A(0)
    method = dlr.second_order_method
    U,S,V,t_vals = vssi.variable_solver(t0,tf,A0,tol,h0,method,k)

    #compute approx soln
    Yt,Ut,St,Vt = vssi.format_Yt(A0,U,S,V)

    #compute truncated SVD soln
    Xt = compute_SVD_t(A,k,t_vals) 

    #compute y_dot for approx soln
    Yt_dot = construct_Y_dot(Ut,St,Vt,t_vals)

    #compute bidiagonal soln
    Wt = construct_Wt(A,k,t_vals)
    
    return Yt,Xt,Yt_dot,Wt,t_vals

def compute_nomrs(t_vals,Xt,Yt,A,Yt_dot,A_dot,Wt):
    # A should be a function
    Xt_AT_array = np.zeros(len(t_vals))
    Yt_AT_array = np.zeros(len(t_vals))
    Xt_YT_array = np.zeros(len(t_vals))
    Yt_AT_dot_array = np.zeros(len(t_vals))
    Wt_AT_array = np.zeros(len(t_vals))
    
    for i,t in enumerate(t_vals):
        Xt_AT_array[i] = np.linalg.norm(Xt[i,:,:]-A(t))
        Yt_AT_array[i] = np.linalg.norm(Yt[i,:,:]-A(t))
        Xt_YT_array[i] = np.linalg.norm(Xt[i,:,:]-Yt[i,:,:])
        Yt_AT_dot_array[i] = np.linalg.norm(Yt_dot[i,:,:]-A_dot(t))
        Wt_AT_array[i] = np.linalg.norm(Wt[i,:,:]-A(t))

    return Xt_AT_array,Yt_AT_array,Xt_YT_array,Yt_AT_dot_array,Wt_AT_array

def plot_norms(t_vals,norm_array1,norm_array2,epsi):
    name = ['Xt_AT','Yt_AT','Xt_YT','Yt_AT_dot','Wt_AT']
    plt.figure(figsize=(12,5))
    plt.suptitle('Norms of the residuals for the rank 10 and 20 solutions, epsilon = '+str(epsi))
    for norm,name in zip(zip(norm_array1,norm_array2),name):
        
        plt.subplot(1,2,1)
        plt.title('rank 10 solutions')
        plt.plot(t_vals,norm[0],label=name)
        plt.legend()
        plt.grid()
        plt.subplot(1,2,2)
        plt.title('rank 20 solutions')
        plt.plot(t_vals,norm[1],label=name)
        plt.legend()
        plt.grid()

    plt.show()
