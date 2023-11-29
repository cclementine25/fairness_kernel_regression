import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import data_generation as dg
import fonctions as fn


def fix_random(seed):
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except:
        pass
    
def train_test_split_groups(x, a, y, random_seed=0, tets_size=0.4):
    """Split the input dataset into train and test sets

    TODO: Need to make sure both train and test sets have enough
    observations from each subgroup
    """
    # size of the training data
    groups = list(a.drop_duplicates())
    x_train_sets = {}
    x_test_sets = {}
    y_train_sets = {}
    y_test_sets = {}
    a_train_sets = {}
    a_test_sets = {}

    for g in groups:
        x_g = x[a == g]
        a_g = a[a == g]
        y_g = y[a == g]
        x_train_sets[g], x_test_sets[g], a_train_sets[g], a_test_sets[g], y_train_sets[g], y_test_sets[g] = train_test_split(x_g, a_g, y_g, test_size=tets_size, random_state=random_seed)

    x_train = pd.concat(x_train_sets.values())
    x_test = pd.concat(x_test_sets.values())
    y_train = pd.concat(y_train_sets.values())
    y_test = pd.concat(y_test_sets.values())
    a_train = pd.concat(a_train_sets.values())
    a_test = pd.concat(a_test_sets.values())

    # resetting the index
    x_train.index = range(len(x_train))
    y_train.index = range(len(y_train))
    a_train.index = range(len(a_train))
    x_test.index = range(len(x_test))
    y_test.index = range(len(y_test))
    a_test.index = range(len(a_test))
    return x_train, a_train, y_train, x_test, a_test, y_test

def subsample(x, a, y, size, random_seed=0):
    """
    Randomly subsample a smaller dataset of certain size
    """
    toss = 1 - size / (len(x))
    x1, _, a1, _, y1 ,_ = train_test_split(x, a, y, test_size=toss, random_state=random_seed)
    x1.index = range(len(x1))
    y1.index = range(len(x1))
    a1.index = range(len(x1))
    return x1, a1, y1

def SMD(y, idx_set):
    smd=0
    for idx in idx_set:
        if  np.sum(idx)==0:
            continue
        smd+=np.abs(np.mean(y[idx])-np.mean(y))
        # print(smd)
    return smd

def gen_idx(s,num_sens, s_0 = 0, s_1 = 1):
    """
    Generate indicators for each sensitive group recursively.
    NOTE: We assume that each attribute has only binary value, i.e. s_0 or s_1. 
    If your data has other values, you should generate the indicator for each sensitive group by yourself.
    """

    idx_1=s[:,0:1]== s_1
    idx_0=s[:,0:1]== s_0
    if num_sens==1:
        return [idx_1, idx_0]
    else:
        other_idx_set=gen_idx(s[:,1:],num_sens-1)
        return [idx_1 * other_idx for other_idx in other_idx_set] + [idx_0 * other_idx for other_idx in other_idx_set]

mu_0 = 2*np.array([-1,-1])#3*np.random.randn(2) 
mu_1 = 2*np.array([1,1])#3*np.random.randn(2) 
L0 = 0.2 * np.array([[ 1.26549968,  0.21514026],
       [-0.17581564, -0.59862356]])
L1 = 0.5 * np.array([[-2.01894502, -0.60109417],
       [-1.30296682, -0.27193436]])
Sigma0 = L0.transpose() @ L0
Sigma1 = L1.transpose() @ L1
Mu = np.array([mu_0,mu_1])
Sigma = np.array([Sigma0,Sigma1])
f_star = lambda x,s : fn.f_star1(x,fn.w)
w = np.array([0.7,0.3])
        
def generate_data(d, n, pri_v = 0.1, f = lambda x : x, noise = 0.1):
    '''
    Function to generate data for the fair regression problem
    d: dimension of the data
    n: number of data points
    pri_v: the value of the privileged attribute
    f: the function to generate the output
    noise: the noise level
    '''
    s_all = dg.gen_S([-1,1], [1/2,1/2], n)
    x_all = dg.gen_X(s_all, lambda s : fn.Om1(s,[-1,1],fn.Mu,fn.Sigma), n)
    z_all =  np.concatenate([np.reshape(x_all,(n,len(x_all[0]))),np.reshape(s_all,(n,len(s_all.shape)))],axis = 1)
    y_all = dg.gen_Y(x_all,s_all, f_star, noise, n)
    x_all = np.random.normal(0, 1, size=[n*2, d])
    return w, z_all, y_all

def record_results(df, model, dataset, y, y_pred, idx_set, y_val, y_val_pred, idx_set_val, DATA_SPLIT_SEED, other_info=None):
    '''
    Function to record the results
    '''
    # Compute metrics
    Train_MSE = np.mean((y-y_pred)**2)
    Test_MSE = np.mean((y_val-y_val_pred)**2)
    Train_SMD = SMD(y_pred, idx_set)
    Test_SMD = SMD(y_val_pred, idx_set_val)
    
    # Print metrics
    print(f'################# Dataset: {dataset}, Method: {model}, DATA_SPLIT_SEED: {DATA_SPLIT_SEED} #################')
    print('Train MSE', Train_MSE)
    print('Test MSE', Test_MSE)
    print('Train SMD', Train_SMD)
    print('Test SMD', Test_SMD)

    # Record results
    row={'Model':model, 'Dataset':dataset, 'Mode':'Train', 'MSE': Train_MSE, 'SMD': Train_SMD, 'DATA_SPLIT_SEED': DATA_SPLIT_SEED}
    if other_info is not None:
        row.update(other_info)    
    df.loc[df.shape[0]]=row
    
    row={'Model':model, 'Dataset':dataset, 'Mode':'Test', 'MSE': Test_MSE, 'SMD': Test_SMD, 'DATA_SPLIT_SEED': DATA_SPLIT_SEED}
    if other_info is not None:
        row.update(other_info)    
    df.loc[df.shape[0]]=row
