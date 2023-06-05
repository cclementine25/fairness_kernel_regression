import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs



def gen_S(S_set,p,n):
    return np.array([np.random.choice(S_set,p = p) for _ in range(n)])
    
def gen_X(S,Om,n):
    return np.array([Om(S[i]) for i in range(n)])
    
def gen_Y(X,S,f_star,sigma,n):
    Eps = sigma * np.random.randn(n)
    return np.array([f_star(X[i],S[i]) for i in range(n)]) + Eps
    
    
    
def shape_S(S):
    if len(S.shape) == 1:
        return 1
    else:
        return len(S[0])

    
