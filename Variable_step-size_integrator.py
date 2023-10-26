import numpy as np
import dynamic_low_rank as dlr 


def variable_solver(t0,tf,U,S,V,A_dot,tol,h0):
    
    count = 0
    t = t0
    while t < tf:
        pass