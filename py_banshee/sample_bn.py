# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 02:46:49 2021
@author: mmendozalugo

"""
import numpy as np
from scipy.interpolate import interp1d
from py_banshee.prediction import ecdf,make_dist
from scipy.stats import norm
import pandas as pd
 
def generate_samples(R,n,names,data,empirical_data=1,distributions=[],parameters=[]):
    '''
    generate_samples compute n samples of the NPBN
    Parameters
    ----------
    R :numpy.ndarray
     Rank Correlation Matrix
    n : int
        numebr of samples to be computed.
    names : list
        list of str with the name of the nodes.
    data : pandas.core.frame.DataFrame
        By default, a matrix containing data for 
        quantifying the NPBN. 
    empirical_data : int,The default is 1.
        1 = DATA is a pd.DataFrame with emprical observations
        if 1 distributions and parameters should be empty i.e. 
        distributions=[], parameters=[]
        0 = Nodes are parametric distributions, no empirical observations available
        if 0 DATA should be empty i.e. DATA = []TYPE, optional 
    distributions :list
        A list with the names of the distributions for each node
        The default is []. 
    parameters : list 
        A list with the corresponding parameters of the distributions
        The default is [].

    Returns
    -------
    samples : pandas.core.frame.DataFrame
        A matrix containing the computed samples for quantifying the NPBN. 

    '''

    d = R.shape[0]     
    if empirical_data ==1 :          
        y =[ecdf(data.iloc[:,i].dropna())[0] for i in range(d)]
        x =[ecdf(data.iloc[:,i].dropna())[1] for i in range(d)]
    else: 
        dists,params = make_dist(distributions,parameters)
        rand_val = [pd.Series(dists[i].rvs(*params[i],n)) for i in range(d)]
        y =[ecdf(rand_val[i])[0] for i in range(d)]
        x =[ecdf(rand_val[i])[1] for i in range(d)]
            
    U = norm.cdf(np.random.multivariate_normal(np.zeros(d),R,n)) #copula rand
    f_int = np.array([interp1d(y[z],x[z],kind='nearest') for z in range(d)])
    samples = np.array([f_int[i](U[:,i]) for i in range(d)]).transpose()
    samples = pd.DataFrame(samples, columns=names)

    return samples


def sample_based_conditioning(data,condition_nodes,LowerBound_UpperBound):
    '''
    sample_based_conditioning computes the explicit based conditioned samples 
    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        Empirical data or generated samles
    condition_nodes : list
        A list with the nodes to be continionalize
    LowerBound_UpperBound : list of list
        A list of lower and upper boud of the base contioning per condition node
        example [[50,100],[850,1250]]

    Returns
    -------
    sample_bc:  pandas.core.frame.DataFrame
        A base contioning samples Matrix

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
    sample_bc = data.iloc[idx_intersec,:].reset_index()
    
    return sample_bc
    

