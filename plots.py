import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs

import data_generation as dg
import predictions as ac
import fonctions as fn 

###### Parameters 

#data number
n_train = 1000
n_test = 10000

#sensitive variable S parameters
S_set = [0,1]
p = [1/2,1/2]
colors = ["skyblue","lightgreen"]

#usual feauture X parameter
Om = lambda s : fn.Om1(s,S_set,fn.Mu,fn.Sigma)

#target Y parameters
f_star = lambda x,s : fn.f_star1(x,fn.w)
sigma = 0.25

#Z
Z_uple = False

#objective function parameters
Gamma = [0,1e-3,1e-2,1e-1,1,10,50,100]
Lambda = [1/(n_train**(1/8)),1e-8,1e-7,1e-10,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1]

#kernels
k = lambda S1,S2 : fn.k1(S1,S2)
m = lambda X1,X2 : fn.m1(X1,X2,1) # here m = g 



#plot parameters

plot_every_hist = True
plot_hist_lambda_opt_for_loss = True
plot_loss_over_gamma_lambda_opt = True
plot_HSIC_over_gamma_lambda_opt = True
HSIC_over_Risk = True
opt_gamma_for_lambda = False



####### Generation and plot of the training data

#generation
S_train = dg.gen_S(S_set, p, n_train)
X_train = dg.gen_X(S_train, Om, n_train)

if Z_uple :
    Z_train =  np.concatenate([np.reshape(X_train,(n_train,len(X_train[0]))),np.reshape(S_train,(n_train,dg.shape_S(S_train)))],axis = 1)
else:
    Z_train = X_train
    
Y_train = dg.gen_Y(X_train,S_train, f_star, sigma, n_train)


#plot

for s in range(len(S_set)):
    Y_S = [Y_train[i]  for i in range(n_train) if S_train[i] == S_set[s]]
    plt.hist(Y_S,bins=30,label = "S = " + str(S_set[s]),color = colors[s])
    plt.plot([np.mean(Y_S),np.mean(Y_S)],[0,n_train/20],"black")
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
    Z_test =  np.concatenate([np.reshape(X_test,(n_test,len(X_test[0]))),np.reshape(S_test,(n_test,dg.shape_S(S_test)))],axis = 1)
else:
    Z_test = X_test
    
Y_test = dg.gen_Y(X_test, S_test, f_star, sigma, n_test)


#plot


plt.figure()
for s in range(len(S_set)):
    Y_S_t = [Y_test[i]  for i in range(n_test) if S_test[i] == S_set[s]]
    plt.hist(Y_S_t,bins=30,label = "S = " + str(S_set[s]),color = colors[s])
    plt.plot([np.mean(Y_S_t),np.mean(Y_S_t)],[0,n_test/10],"black")
    plt.legend()
    plt.axis([np.min(Y_test),np.max(Y_test),0,n_test/10])
    plt.title("repartition of the labels for s=0 and s=1" )

#plt.figure()
#plt.hist(Y_test,bins = 30)
#plt.title("repartition of the labels")
#plt.axis([np.min(Y_test),np.max(Y_test),0,n_test/20])



##### Computation of the predictions

K_tt = ac.K_mat(S_test,S_train,k)
M_tt = ac.M_mat(X_test,X_train,m,K_tt,Z_uple)

Pred = ac.preds_comp(Alpha,M_tt,Lambda,Gamma)

K_test = ac.K_mat(S_test,S_test,k)
I_test = np.identity(n_test)
H_test = I_test - 1/n_test * np.ones(n_test)
B_test =  H_test @ K_test @ H_test 
Loss = ac.Loss_f(Pred,Y_test,Gamma,Lambda)
HSIC = ac.HSIC_f(Pred,B_test,n_test,Gamma,Lambda)


##### Plots of the predictions

#plot of all the histograms of the grid search


if plot_every_hist :
    fig, axs = plt.subplots(len(Lambda), len(Gamma), figsize=(20,20))
    for i in range(len(Lambda)):
        for j in range(len(Gamma)):
            for s in range(len(S_set)):
                Y_S_pred = [Pred[i][j][k]  for k in range(n_test) if S_test[k] == S_set[s]]
                axs[i,j].hist(Y_S_pred,bins=60,label = "S =" + str(S_set[s]),color = colors[s]) 
                axs[i,j].plot([np.mean(Y_S_pred),np.mean(Y_S_pred)],[0,n_test/20],"black")
            axs[i,j].legend()
            axs[i,j].set_title(r"$\lambda = $" + str(Lambda[i]) + r"$\gamma = $" + str(Gamma[j]))
            axs[i,j].axis([np.min(Y_test),np.max(Y_train),0,n_test/20])
            axs[i,j].get_xaxis().set_visible(False)
            #print(4*i + j)

    fig.suptitle('Repartition of the predictions for s=0 and s=1')

    plt.show()
  
    
#plot the histograms for differents values of gamma of the repartition of the labels for the optimal value of lambda in term of loss
Ind_l =[np.argmin(Loss[:,j]) for j in range(len(Gamma))]
Ind_l_2 =  [0 for j in range(len(Gamma))] # #[0 for j in range(len(Gamma))]

if plot_hist_lambda_opt_for_loss :
    fig, axs = plt.subplots(len(Gamma)//4 , 4, figsize=(25,8))
    for j in range(len(Gamma)):
        i = Ind_l[j]
        a = j//4
        b = j%4
        for s in range(len(S_set)):
            Y_S_pred = [Pred[i][j][k]  for k in range(n_test) if S_test[k] == S_set[s]]
            axs[a,b].hist(Y_S_pred,bins=60,label = "S = " + str(S_set[s]),color = colors[s])
            axs[a,b].plot([np.mean(Y_S_pred),np.mean(Y_S_pred)],[0,300],"black")
        axs[a,b].legend()
        axs[a,b].set_title(r"$\gamma = $" + str(Gamma[j]) + r" $\lambda_{opt} = $  " +  str(Lambda[Ind_l[j]]))
        axs[a,b].axis([np.min(Y_test),np.max(Y_test),0,300])
    
    fig.suptitle(r'Repartition of the predictions for s=0 and s=1 for fifferent values of $\gamma$ and optimal $\lambda$')
    
    plt.show()    
    
#Plot of the loss over gamma with opt lambda
if plot_loss_over_gamma_lambda_opt:
    plt.figure(figsize=(20,7))
    plt.plot(Gamma, [Loss[ Ind_l[j],j] for j in range(len(Gamma))], label = r"$\lambda = \lambda_{opt}(\gamma)$")
    plt.scatter(Gamma, [Loss[ Ind_l[j],j] for j in range(len(Gamma))])
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Loss")
    plt.title(r"Evolution of the Loss with $\gamma$ for an optimal $\lambda_{\gamma}$")

    plt.plot(Gamma, [Loss[ Ind_l_2[j],j] for j in range(len(Gamma))], label = r"$\lambda = \lambda_{opt}^r$")
    plt.scatter(Gamma, [Loss[ Ind_l_2[j],j] for j in range(len(Gamma))])
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Loss")
    plt.title(r"Evolution of the Loss with $\gamma$ for an optimal $\lambda_{\gamma}$")  
    
plt.legend()

#Plot of HSIC over gamma with opt lambda
if plot_HSIC_over_gamma_lambda_opt:
    plt.figure(figsize=(20,7))
    plt.plot(Gamma, [HSIC[Ind_l[j],j] for j in range(len(Gamma))], label =r"$\lambda = \lambda_{opt}(\gamma)$")
    plt.scatter(Gamma, [HSIC[Ind_l[j],j] for j in range(len(Gamma))])
    plt.xlabel(r"$\gamma$")
    plt.ylabel("HSIC")
    plt.title(r"Evolution of the HSIC with $\gamma$ for an optimal $\lambda_{\gamma}$")

    plt.plot(Gamma, [HSIC[Ind_l_2[j],j] for j in range(len(Gamma))],label = r"$\lambda = \lambda_{opt}^r$")
    plt.scatter(Gamma, [HSIC[Ind_l_2[j],j] for j in range(len(Gamma))])
    plt.xlabel(r"$\gamma$")
    plt.ylabel("HSIC")
    plt.title(r"Evolution of the HSIC with $\gamma$ for an optimal $\lambda_{\gamma}$")

plt.legend()


plt.figure()
plt.plot(Gamma,[np.linalg.norm(Alpha[0][j]) for j in range(len(Gamma))])
plt.title(r"norme de $\alpha$")


#Plot HSIC over the risk
Ind_g = [np.argmin(HSIC[i,:]) for i in range(len(Lambda))]
plt.figure() 
if HSIC_over_Risk :
    plt.scatter([Loss[Ind_l[j],j] for j in range(len(Gamma))],[HSIC[Ind_l[j],j] for j in range(len(Gamma))])
    plt.plot([Loss[Ind_l[j],j] for j in range(len(Gamma))],[HSIC[Ind_l[j],j] for j in range(len(Gamma))])
    plt.scatter([Loss[i,Ind_g[i]] for i in range(len(Lambda))], [HSIC[i,Ind_g[i]] for i in range(len(Lambda))],color = "red")
    plt.plot([Loss[i,Ind_g[i]] for i in range(len(Lambda))], [HSIC[i,Ind_g[i]] for i in range(len(Lambda))],color = "red")
plt.xlabel(r"Loss")
plt.ylabel("HSIC")


#plot optimal gamma for lambda
if opt_gamma_for_lambda:
    fig, axs = plt.subplots(len(Lambda)//4 , 4, figsize=(25,8))
    for i in range(len(Lambda)):
        j = Ind_g[i]
        a = i//4
        b = i%4
        for s in range(len(S_set)):
            Y_S_pred = [Pred[i][j][k]  for k in range(n_test) if S_test[k] == S_set[s]]
            axs[a,b].hist(Y_S_pred,bins=60,label = "S = " + str(S_set[s]))
            axs[a,b].plot([np.mean(Y_S_pred),np.mean(Y_S_pred)],[0,20],"black")
        axs[a,b].legend()
        axs[a,b].set_title(r"$\lambda = $" + str(Lambda[i]) + r" $\gamma_{opt} = $  " +  str(Gamma[Ind_g[i]]))
        axs[a,b].axis([np.min(Y_test),np.max(Y_test),0,20])
    
    fig.suptitle(r'Repartition of the predictions for s=0 and s=1 for fifferent values of $\gamma$ and optimal $\lambda$')
    
    plt.show()    













    
    
    
    
    
    
    