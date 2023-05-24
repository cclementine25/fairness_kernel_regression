import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs


#kernel matrices 

def K_mat(S1,S2,k):
    return k(S1,S2)

def M_mat(X1,X2,m,K,Z_uple):
    M = m(X1,X2)
    if Z_uple:
        return M * K
    else:
        return M
  
#useful functions

def A_f(lambd,gamma,M,C,Y,n):
    D = lambd * n * M +  M @ M + gamma/n * C
    return np.linalg.pinv(D) @ M @ Y

def f_reg(alpha,K_test):
    return np.dot(alpha,K_test) 

def hSIC_f(Pred,B,n):
    return Pred.reshape(1,n) @ B @ Pred.reshape(n,1)

def loss_f(Y_test,Y_pred):
    return np.mean((Y_test - Y_pred)**2)

def av_dif_f(Pred,S_test,n):
    Pred_0 = np.array([Pred[i]  for i in range(n) if S_test[i] == 0])
    Pred_1 = np.array([Pred[i]  for i in range(n) if S_test[i] == 1])
    return np.abs(np.mean(Pred_0) - np.mean(Pred_1)) 


#representation vector computation for different values of alpha and gamma   
 
    
def alpha_comp(X,S,k,m,Lambda,Gamma,Z_uple,M,C,Y,n):
    Alpha = np.zeros(((len(Lambda),len(Gamma),len(X))))
    for l in range(len(Lambda)):
        for g in range(len(Gamma)):
            Alpha[l][g] = A_f(Lambda[l],Gamma[g],M,C,Y,n)
    
    return Alpha
    
# prediction for different values of lambda and gamma


def preds_comp(Alpha,K_test,Lambda,Gamma):
    Pred = np.array([[K_test @ Alpha[i][j] for j in range(len(Gamma))] for i in range(len(Lambda))])
    return Pred
    
    
    
# loss, HSIC and average difference computation for lambda,gamma  

def Loss_f(Pred,Y_test,Gamma,Lambda):
    Loss = np.zeros((len(Lambda),len(Gamma)))
    for l in range(len(Lambda)):
        for g in range(len(Gamma)):
            Loss[l][g] = loss_f(Pred[l][g],Y_test)
    return Loss

def HSIC_f(Pred,B,n,Gamma,Lambda):
    HSIC = np.zeros((len(Lambda),len(Gamma)))
    for l in range(len(Lambda)):
        for g in range(len(Gamma)):
            HSIC[l][g] = hSIC_f(Pred[l][g],B,n)
    return HSIC


    
    
    
