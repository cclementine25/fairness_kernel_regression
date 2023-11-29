import numpy as np
import scipy.stats as scs


##########################################################
#### Variables

mu_0 = 2*np.array([-1,-1])#3*np.random.randn(2) 
mu_2 = 2*np.array([1,1])#3*np.random.randn(2) 
mu_1 = np.array([0,0])
L1 = 0.2 * np.array([[ 1.26549968,  0.21514026],
       [-0.17581564, -0.59862356]])
L0 = 0.4 * np.array([[-2.01894502, -0.60109417],
       [-1.30296682, -0.27193436]])
L2 = np.array([[-0.59893527,  0.4090206 ],
       [ 0.2676863 , -0.49630348]])
Sigma0 = L0.transpose() @ L0
Sigma1 = L1.transpose() @ L1
Sigma2 = L2.transpose() @ L2


Mu = np.array([mu_0,mu_1])
Sigma = np.array([Sigma0,Sigma1])

Mu_3s = np.array([mu_0,mu_1,mu_2])
Sigma_3s = np.array([Sigma0,Sigma1,Sigma2])

w = np.array([0.7,0.3])

BB = 0.1 * np.ones((2,2))

##########################################################"
mu_00 = 2*np.array([-1,-1,-1])#3*np.random.randn(2) 
mu_11 = 2*np.array([1,1,1])
L00 = 0.2 * np.random.randn(3,3)
L11 = 0.5 * np.random.randn(3,3)
Sigma00 = L00.transpose() @ L00
Sigma11 = L11.transpose() @ L11
ww = np.array([0.7,0.3,0.5])
Muu =np.array([mu_00,mu_11])
Sigmaa = np.array([Sigma00,Sigma11]) 


################################################




Mu = np.array([mu_0,mu_1])
Sigma = np.array([Sigma0,Sigma1])

############################################################
#### Kernels K

def id(s1,s2):
    if s1 == s2:
        return 1
    return 0

def k1(S1,S2):
    n1 = len(S1)
    n2 = len(S2)
    S1 = S1.reshape(n1,1) * np.ones(n2)
    S2 = S2.reshape(1,n2) * np.ones(n1).reshape(n1,1)
    return  1 + 2 * (S1*S2) - (S1 + S2)

def k2(S1,S2):
    n1,n2 = len(S1),len(S2)
    return S1.reshape(n1,1) * S2.reshape(1,n2)

def k3(S1,S2):
    n1 = len(S1)
    n2 = len(S2)
    return np.array([[id(s1,s2) for s2 in S2] for s1 in S1])
    



#############################################################
#### Kernels M (for X)

def m1(X1,X2,sigma):
    n1 = len(X1)
    n2 = len(X2)
    Diff = X1.reshape(n1,1,len(X1[0])) - X2.reshape(1,n2,len(X1[0]))
    G = np.exp(-np.linalg.norm(Diff,axis = 2)**2 / (2*sigma))
    return G 

def m2(X1,X2):
    return np.array([[np.dot(x1,x2) for x2 in X2] for x1 in X1])

def m3(X1,X2):
    return np.array([[x1[0]*x2[0] + x1[1]*x2[0] for x2 in X2] for x1 in X1])

def m4(X1,X2):
    return np.array([[np.min(x1 + x2) for x2 in X2] for x1 in X1])
    


##############################################################
### X = Om(S)

def Om1(s, S_set, Mu,Sigma):
    i = S_set.index(s)
    mu = Mu[i]
    sigm = Sigma[i]
    return scs.multivariate_normal.rvs(mu,sigm)



#############################################################
### Y = f_star

def f_star1(X,w):
    return np.dot(w,X)


def f_star2(x,B,w):
    return np.dot(x,B @ x) + np.dot(x,w)

def f_star3(x,w):
    return np.exp(np.dot(x,w))

def f_star4(x,s,w):
    return np.cos(np.dot(x,w)) + 3 * s

def f_star5(X,S,w):
    return np.dot(w,X)



    