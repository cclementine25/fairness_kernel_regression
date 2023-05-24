import numpy as np
import scipy.stats as scs


#### Variables

mu_0 = np.array([-1,-1])#3*np.random.randn(2) 
mu_1 = np.array([1,1])#3*np.random.randn(2) 
L0 = 0.5 * np.array([[ 0.44802685,  2.1254507 ],
       [ 1.00050239, -1.12101325]])
L1 = 0.5 * np.array([[-2.01894502, -0.60109417],
       [-1.30296682, -0.27193436]])
Sigma0 = L0.transpose() @ L0
Sigma1 = L1.transpose() @ L1


Mu = np.array([mu_0,mu_1])
Sigma = np.array([Sigma0,Sigma1])

#### Kernels K

def k1(S1,S2):
    n1 = len(S1)
    n2 = len(S2)
    S1 = S1.reshape(n1,1) * np.ones(n2)
    S2 = S2.reshape(1,n2) * np.ones(n1).reshape(n1,1)
    return  1 + 2 * (S1*S2) - (S1 + S2)



#### Kernels M (for X)

def m1(X1,X2,sigma):
    n1 = len(X1)
    n2 = len(X2)
    Diff = X1.reshape(n1,1,2) - X2.reshape(1,n2,2)
    G = np.exp(-np.linalg.norm(Diff,axis = 2)**2 / (2*sigma))
    return G 


### X = Om(S)

def Om1(s, S_set, Mu,Sigma):
    i = S_set.index(s)
    mu = Mu[i]
    sigm = Sigma[i]
    return scs.multivariate_normal.rvs(mu,sigm)

### Y = f_star

def f_star1(X,w):
    return np.dot(w,X)
    