import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs

import data_generation as dg
import predictions as ac
import fonctions as fn 

###### Parameters 

#data number
n_train = 1000
n_test = 1000

#sensitive variable S parameters
S_set = [0,1]
p = [1/2,1/2]

#usual feauture X parameter
Om = lambda s : fn.Om1(s,S_set,fn.Mu,fn.Sigma)

#target Y parameters
f_star = lambda x,s : fn.f_star1(x,np.array([1,1/2]))
sigma = 0.25

#Z
Z_uple = True

#objective function parameters
Gamma = [0,1e-5,1e-4,1e-3,1e-2,1e-1,1,10]
Lambda = [0,1e-5,1e-4,1e-3,1e-2,1e-1,1,10]

#kernels
k = lambda S1,S2 : fn.k1(S1,S2)
m = lambda X1,X2 : fn.m1(X1,X2,1)



#plot parameters

plot_every_hist = True
plot_hist_lambda_opt_for_loss = True
plot_loss_over_gamma_lambda_opt = True
plot_HSIC_over_gamma_lambda_opt = True



####### Generation and plot of the training data

#generation
S_train = dg.gen_S(S_set, p, n_train)
X_train = dg.gen_X(S_train, Om, n_train)

if Z_uple :
    Z_train =  np.concatenate([np.reshape(X_train,(n_train,len(X_train[0]))),np.reshape(S_train,(n_train,len(S_train.shape)))],axis = 1)
else:
    Z_train = X_train
    
Y_train = dg.gen_Y(X_train,S_train, f_star, sigma, n_train)


#plot

Y_0 = [Y_train[i]  for i in range(n_train) if S_train[i] == 0]
Y_1 = [Y_train[i]  for i in range(n_train) if S_train[i] == 1]
plt.hist(Y_0,color ="red",bins=30,label = "S = 0")
plt.hist(Y_1,color ="blue",bins=30,label = "S = 1")
plt.plot([np.mean(Y_0),np.mean(Y_0)],[0,n_train/20],"black")
plt.plot([np.mean(Y_1),np.mean(Y_1)],[0,n_train/20],"black")
plt.legend()
plt.axis([np.min(Y_train),np.max(Y_train),0,n_train/20])
plt.title("repartition of the labels for s=0 and s=1" )

#plt.figure()
#plt.hist(Y_train,bins = 30)
#plt.title("repartition of the labels")
#plt.axis([np.min(Y_train),np.max(Y_train),0,n_train/20])


#kernels matrices 

I = np.identity(n_train)
H = I - 1/n_train * np.ones(n_train)

K = ac.K_mat(S_train,S_train,k)
M = ac.M_mat(X_train,X_train,m,K,Z_uple)

B = H @ K @ H 
C = M @ B @ M



##### Computation of the representation vector


Alpha = ac.alpha_comp(X_train,S_train,k,m,Lambda,Gamma,Z_uple,M,C,Y_train,n_train)


#### Generation and plot of test data

#generation
S_test = dg.gen_S(S_set, p, n_test)
X_test = dg.gen_X(S_test, Om, n_test)

if Z_uple :
    Z_test =  np.concatenate([np.reshape(X_test,(n_test,len(X_test[0]))),np.reshape(S_test,(n_test,len(S_test.shape)))],axis = 1)
else:
    Z_test = X_test
    
Y_test = dg.gen_Y(X_test, S_test, f_star, sigma, n_test)


#plot

plt.figure()
Y_0_t = [Y_test[i]  for i in range(n_test) if S_test[i] == 0]
Y_1_t = [Y_test[i]  for i in range(n_test) if S_test[i] == 1]
plt.hist(Y_0_t,color ="red",bins=30,label = "S = 0")
plt.hist(Y_1_t,color ="blue",bins=30,label = "S = 1")
plt.plot([np.mean(Y_0_t),np.mean(Y_0_t)],[0,n_test/20],"black")
plt.plot([np.mean(Y_1_t),np.mean(Y_1_t)],[0,n_test/20],"black")
plt.legend()
plt.axis([np.min(Y_test),np.max(Y_test),0,n_test/20])
plt.title("repartition of the labels for s=0 and s=1" )

#plt.figure()
#plt.hist(Y_test,bins = 30)
#plt.title("repartition of the labels")
#plt.axis([np.min(Y_test),np.max(Y_test),0,n_test/20])



##### Computation of the predictions

K_test = ac.K_mat(S_test,S_train,k)
M_test = ac.M_mat(X_test,X_train,m,K,Z_uple)

Pred = ac.preds_comp(Alpha,M_test,Lambda,Gamma)

Loss = ac.Loss_f(Pred,Y_test,Gamma,Lambda)
HSIC = ac.HSIC_f(Pred,B,n_test,Gamma,Lambda)


##### Plots of the predictions

#plot of all the histograms of the grid search


if plot_every_hist :
    fig, axs = plt.subplots(len(Lambda), len(Gamma), figsize=(20,20))
    for i in range(len(Lambda)):
        for j in range(len(Gamma)):
            Y_0_pred = [Pred[i][j][k]  for k in range(n_test) if S_test[k] == 0]
            Y_1_pred = [Pred[i][j][k]  for k in range(n_test) if S_test[k] == 1]
            axs[i,j].hist(Y_0_pred,color ="red",bins=60,label = "S = 0")
            axs[i,j].hist(Y_1_pred,color ="blue",bins=60,label = "S = 1")
            axs[i,j].plot([np.mean(Y_0_pred),np.mean(Y_0_pred)],[0,n_test/20],"black")
            axs[i,j].plot([np.mean(Y_1_pred),np.mean(Y_1_pred)],[0,n_test/20],"black")
            axs[i,j].legend()
            axs[i,j].set_title(r"$\lambda = $" + str(Lambda[i]) + r"$\gamma = $" + str(Gamma[j]))
            axs[i,j].axis([np.min(Y_test),np.max(Y_train),0,n_test/20])
            #print(4*i + j)

    fig.suptitle('Repartition of the predictions for s=0 and s=1')

    plt.show()
  
    
#plot the histograms for differents values of gamma of the repartition of the labels for the optimal value of lambda in term of loss
Ind_l = [np.argmin(Loss[:,j]) for j in range(len(Gamma))]

if plot_hist_lambda_opt_for_loss :
    fig, axs = plt.subplots(len(Gamma)//4 , 4, figsize=(25,8))
    for j in range(len(Gamma)):
        i = Ind_l[j]
        Y_0_pred = [Pred[i][j][k]  for k in range(n_test) if S_test[k] == 0]
        Y_1_pred = [Pred[i][j][k]  for k in range(n_test) if S_test[k] == 1]
        a = j//4
        b = j%4
        axs[a,b].hist(Y_0_pred,color ="red",bins=60,label = "S = 0")
        axs[a,b].hist(Y_1_pred,color ="blue",bins=60,label = "S = 1")
        axs[a,b].plot([np.mean(Y_0_pred),np.mean(Y_0_pred)],[0,20],"black")
        axs[a,b].plot([np.mean(Y_1_pred),np.mean(Y_1_pred)],[0,20],"black")
        axs[a,b].legend()
        axs[a,b].set_title(r"$\gamma = $" + str(Gamma[j]) + r" $\lambda_{opt} = $  " +  str(Lambda[Ind_l[j]]))
        axs[a,b].axis([-10,10,0,20])
    
    fig.suptitle(r'Repartition of the predictions for s=0 and s=1 for fifferent values of $\gamma$ and optimal $\lambda$')
    
    plt.show()    
    
#Plot of the loss over gamma with opt lambda
if plot_loss_over_gamma_lambda_opt:
    plt.figure(figsize=(20,7))
    plt.plot(Gamma, [Loss[j, Ind_l[j]] for j in range(len(Gamma))])
    plt.scatter(Gamma, [Loss[j, Ind_l[j]] for j in range(len(Gamma))])
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Loss")
    plt.title(r"Evolution of the Loss with $\gamma$ for an optimal $\lambda_{\gamma}$")

    

#Plot of HSIC over gamma with opt lambda
if plot_HSIC_over_gamma_lambda_opt:
    plt.figure(figsize=(20,7))
    plt.plot(Gamma, [HSIC[j, Ind_l[j]] for j in range(len(Gamma))])
    plt.scatter(Gamma, [HSIC[j, Ind_l[j]] for j in range(len(Gamma))])
    plt.xlabel(r"$\gamma$")
    plt.ylabel("HSIC")
    plt.title(r"Evolution of the HSIC with $\gamma$ for an optimal $\lambda_{\gamma}$")

