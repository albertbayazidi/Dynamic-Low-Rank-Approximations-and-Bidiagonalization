# test av oppgave 3
tol = 1.e-5
t0 = 0
tf = 0.02
h0 = 0.04 # initial step size in vssi
N = 32
k = 1 # k- rank approx

# def grid
x = np.linspace(0,1,N)
y = np.linspace(0,1,N)
X,Y = np.meshgrid(x,y)


# def exact soln
u_ex_f = u_exact(X,Y,tf)

# Initialize u and u_dot
m,n = N,N
u0 = tilr.u_fun(g,m,n) 

# specify what method to use
method = tilr.second_order_method

# do method

U,S,V,t_vals = vssi.variable_solver(t0,tf,u0,tol,h0,method,k)

Yt,Ut,St,Vt = vssi.format_Yt(u0,U,S,V)

plt.imshow(Yt[-1,:,:])
plt.colorbar()
