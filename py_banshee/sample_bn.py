# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 02:46:49 2021
Sample BN olnly for empirical data
@author: mmendozalugo
"""
import numpy as np
from scipy.interpolate import interp1d
from py_banshee.prediction import ecdf
from scipy.stats import norm
import pandas as pd
 
def generate_samples(data,R,n):
    '''
    
    Parameters
    ----------
    data : TYPE: pandas.core.frame.DataFrame
        DESCRIPTION. Empirical data
    R : TYPE: numpy.ndarray
        DESCRIPTION. Rank correlation matrix
    n : TYPE. int
        DESCRIPTION. Number of samples 

    Returns
    -------
    samples : TYPE: pandas.core.frame.DataFrame
        DESCRIPTION. generated samples.

    '''
    d = R.shape[0]               
    y =[ecdf(data.iloc[:,i])[0] for i in range(d)]
    x =[ecdf(data.iloc[:,i])[1] for i in range(d)]
    
    U = norm.cdf(np.random.multivariate_normal(np.zeros(d),R,n)) #copula rand
    f_int = np.array([interp1d(y[z],x[z],kind='nearest') for z in range(d)])
    samples = np.array([f_int[i](U[:,i]) for i in range(d)]).transpose()
    samples = pd.DataFrame(samples, columns=data.columns)

    return samples


def sample_base_conditioning(data,condition_nodes,LowerBound_UpperBound):
    '''
    
    Parameters
    ----------
    data : TYPE: pandas.core.frame.DataFrame
        DESCRIPTION. Empirical data
    condition_nodes : TYPE: list
        DESCRIPTION. nodes to be continionalize
    LowerBound_UpperBound : TYPE. list of touples
        DESCRIPTION. lower and upper boud of the base contioning per condition node

    Returns
    -------
    sample_bc: TYPE: pandas.core.frame.DataFrame
        DESCRIPTION. base contioning samples

    '''
    data_cond = data.iloc[:,condition_nodes]  #conditionalized variables 
   
    #finding the observations between LowerBound_UpperBound (lb-ub)
    def data_between(data,lb_ub):
        #db=[]
        idx=[]
        for i in range(len(data)):
            if data[i]>=lb_ub[0] and data[i]<=lb_ub[1]:
                idx.append(i)
                #db.append(data[i])
        return idx
    
    def intersection(*lists):
        return set(lists[0]).intersection(*lists[1:])
    
    #index of the observations between lb-ub
    idx=[data_between(data_cond.iloc[:,i].tolist(),LowerBound_UpperBound[i]) for i in range(data_cond.shape[1])]
    #index that have all conditions
    idx_intersec = list(intersection(*idx))
    #sample base conditioning
    sample_bc = data.iloc[idx_intersec,:]
    
    return sample_bc
    

