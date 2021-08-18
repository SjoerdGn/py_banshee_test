# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 11:04:45 2021
@author: Paul

Modified on Mon Jul 19 21:28:26 2021
@author: mmendozalugo
"""

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from py_banshee.rankcorr import ranktopearson
from scipy.stats import norm
from scipy.interpolate import interp1d


def list_dif(li1, li2):
    return (list(list(set(li1)-set(li2)) ))

def inference(Nodes, Values, R, DATA, Output='full', 
                SampleSize=1000, Interp='next',
                empirical_data=1, distributions=[],parameters=[]):

    # Defining nodes to be predicted and their number
    remaining_nodes = list_dif(list(range(R.shape[0])),Nodes)
    nr_remaining_nodes = len(remaining_nodes)
    
    if type(Values)==list:
        Values = np.array(Values).reshape(1,-1)
        
    n_values=np.shape(Values)[0]
    
    if Output=='full':
        F = np.zeros((n_values,nr_remaining_nodes,SampleSize))
    else:
        F = np.zeros((n_values,nr_remaining_nodes))
    
    # Adding additional edge cases for the marginal empirical distributions
    # in order to avoid NaN values when conditionalizing
    if empirical_data==1:
        m1 = (DATA.min()-0.1).to_frame().T
        m2 = (DATA.max()+0.1).to_frame().T
        
        DATA=pd.concat([m2, DATA, m1]).reset_index(drop=True)
    else:
        DATA=[]

    
    # # Obtaining the number of nodes (according to the correlation matrix R)
    n_nodes = np.shape(R)[0]  
    
    if empirical_data==1:
        count=0
        x=[None]*n_nodes
        f=[None]*n_nodes
        for node in DATA:
            f[count],x[count]=ecdf(DATA[node])
            count+=1
    else:
        dists,params = make_dist(distributions,parameters)
            
    if empirical_data==0:
        if len(distributions)!= n_nodes:
            raise Exception('Please check the distributions and parameters')
        
    # Transforming Spearman's rank correlation into Pearson's correlation 
    # (auxiliary function 1)
    rpearson=ranktopearson(R)
       
    # Loop for inference for each row in VALUES
    for j in range(n_values):
        # Obtaining the conditional inverse normal distributions at each node
        NormalCond=np.zeros(len(Nodes))  # preallocation

        # for i in Nodes:
        if empirical_data==1:    
            for i in range(len(Nodes)):
                # Create index i_nodes, who points to the correct column within
                # lists x and f. Necessary when columns to predict are not the last
                # columns in DATA.
                i_nodes=Nodes[i]
                x_int=[x[i_nodes][0]-(x[i_nodes][1]-x[i_nodes][0])]+x[i_nodes][1:]
                y_int=[0]+f[i_nodes][1:]
                f_int=interp1d(x_int,y_int)
                NormalCond[i] = norm.ppf(f_int(Values[j,i]))
        else:
            for i in range(len(Nodes)):
                NormalCond[i] = norm.ppf(dists[i].cdf(Values[j,i],*params[i]))
       
            
        # Calculating the parameters of the conditional normal distribution 
        # (auxiliary function 2)
        M_c,S_c = ConditionalNormal(np.zeros(n_nodes),rpearson,Nodes,NormalCond)
        
        # Sometimes S_c just fails the symmetry test because S_c' differs 
        # slightly from S_c due to numerical errors. Therefore, S_c is 
        # symmetrized in the next step:
        S_c_symm = (S_c+S_c.transpose())/2
        
        # Sampling the conditional normal distribution
        norm_samples = np.random.multivariate_normal(M_c,S_c_symm,SampleSize)
        
        # Extracting values of the empirical marginal distributions using the
        # probability density function of the conditional normal distribution.
        # The calculation uses auxiliary function 3
        F0=[None]*nr_remaining_nodes #preallocation
        
        for i in range(nr_remaining_nodes):
            if empirical_data==1:
                F0[i] = inv_empirical(norm.cdf(norm_samples[:,i]),
                                  [f[remaining_nodes[i]],x[remaining_nodes[i]]],
                                  Interp)
            else:
                 F0[i] = dists[remaining_nodes[i]].ppf(norm.cdf(norm_samples[:,i]),*params[remaining_nodes[i]])
        
        if Output=='full':
            for i in range(nr_remaining_nodes):
                F[j,i,:]=F0[i]
        elif Output=='mean':
            for i in range(nr_remaining_nodes):
                F[j,i]=np.mean(F0[i])
        
        if n_values>100:
            txt = 'Making inference. Progress: '
            prog = np.floor(j/n_values*100)
            print('%s %d%%' %(txt, prog))
                   
    return F

# -------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
# -------------------------------------------------------------------------
#
# -------------------------------------------------------------------------
# 1 - Calculating the marginal Empirical Cumulative Distribution Function
# -------------------------------------------------------------------------
def ecdf(column):
    sq=column.value_counts()
    a=sq.sort_index().cumsum()*1./len(column)
    f=a.tolist()
    x=a.index.values.tolist()  
    # Add a starting point to ecdf to make ecdf start at f=0 
    # (Corresponds with MATLAB ecdf function)
    f=[0]+f
    x=[x[0]]+x
    return f,x

# -------------------------------------------------------------------------
# 2 - Calculating the parameters of the conditional normal distribution
# -------------------------------------------------------------------------
def ConditionalNormal(M, S, idxC, valC):
    # Mc: mean vector of the conditional multivariate normal on valC
    # Sc: covariance matrix of the conditional multivariate normal valC

    # M: mean vector of the multivariate normal
    # S: covariance matrix of the multivariate normal
    # idxC: index of the conditioning nodes
    # valC: values of the conditioning nodes 
    
    D = len(M)                          # Dimension of the multivariate normal
    idxNC = list_dif(range(D),idxC)     # Index of all the remaining variables

    # Calculation of the conditional normal distribution:
    M1 = M[idxNC]
    S11 = S[np.ix_(idxNC,idxNC)]
    X2 = valC
    M2 = M[idxC]
    S22 = S[np.ix_(idxC, idxC)]
    S12 = S[np.ix_(idxNC, idxC)]
    S21 = S[np.ix_(idxC, idxNC)]
    S22_inv = np.linalg.inv(S22)
    
    Sc = S11 - S12 @ S22_inv @ S21
    Mc = M1 + S12 @ S22_inv @ (X2-M2)
    return Mc, Sc

# -------------------------------------------------------------------------
# 3 - Calculating the inverse of the conditional empirical distribution
# -------------------------------------------------------------------------
def inv_empirical(yi, empcdf, way):
    # xi:      inverse of the conditional empirical distribution
    
    # yi:      the samples of the conditional normal distribution
    # empcdf:  the empirical marginal distribution of a given node
    # way:     interpolation method in interp1 function ('next' is default)
    
    fe = empcdf[0]                    # cumulative probability density
    xe = empcdf[1]                    # corresponding empirical values

    func_i = interp1d(fe,xe,kind=way)   # interpolation
    
    xi = func_i(yi)
    return xi

# -------------------------------------------------------------------------
# 4 - Parametric distributions
# -------------------------------------------------------------------------


def make_dist(distributions,parameters):
    '''
    

    Parameters
    ----------
    distributions : TYPE list
        DESCRIPTION. List of the names of the distribtions fro each node 
    parameters : TYPE list
        DESCRIPTION. list of the corresponding parameters of the distributions

    Returns
    -------
    dists : TYPE
        DESCRIPTION. scipy distributions objects
    params : TYPE
        DESCRIPTION. parameters of the distributions 

    '''   
    dist_all =[scipy.stats.alpha,scipy.stats.anglit, scipy.stats.arcsine, scipy.stats.argus,scipy.stats.beta,
        scipy.stats.betaprime,scipy.stats.bradford,scipy.stats.burr, scipy.stats.burr12, scipy.stats.cauchy,
        scipy.stats.chi, scipy.stats.chi2, scipy.stats.cosine, scipy.stats.crystalball, scipy.stats.dgamma,
        scipy.stats.dweibull,  scipy.stats.erlang, scipy.stats.expon, scipy.stats.exponnorm, scipy.stats.exponweib,
        scipy.stats.exponpow, scipy.stats.f, scipy.stats.fatiguelife, scipy.stats.fisk,  scipy.stats.foldcauchy,
        scipy.stats.foldnorm, scipy.stats.genlogistic, scipy.stats.gennorm,  scipy.stats.genpareto,scipy.stats.genexpon,
        scipy.stats.genextreme, scipy.stats.gausshyper, scipy.stats.gamma, scipy.stats.gengamma, scipy.stats.genhalflogistic,
        scipy.stats.geninvgauss,  scipy.stats.gilbrat,  scipy.stats.gompertz, scipy.stats.gumbel_r, scipy.stats.gumbel_l,
        scipy.stats.halfcauchy,  scipy.stats.halflogistic, scipy.stats.halfnorm,  scipy.stats.halfgennorm,
        scipy.stats.hypsecant, scipy.stats.invgamma, scipy.stats.invgauss, scipy.stats.invweibull,  scipy.stats.johnsonsb,
        scipy.stats.johnsonsu, scipy.stats.kappa4, scipy.stats.kappa3, scipy.stats.ksone, scipy.stats.kstwo,
        scipy.stats.kstwobign,  scipy.stats.laplace, scipy.stats.laplace_asymmetric, scipy.stats.levy,
        scipy.stats.levy_l, scipy.stats.levy_stable, scipy.stats.logistic, scipy.stats.loggamma,
        scipy.stats.loglaplace,scipy.stats.lognorm, scipy.stats.loguniform, scipy.stats.lomax,
        scipy.stats.maxwell, scipy.stats.mielke, scipy.stats.moyal, scipy.stats.nakagami, scipy.stats.ncx2,
        scipy.stats.ncf, scipy.stats.nct, scipy.stats.norm, scipy.stats.norminvgauss, scipy.stats.pareto,
        scipy.stats.pearson3, scipy.stats.powerlaw, scipy.stats.powerlognorm, scipy.stats.powernorm,
        scipy.stats.rdist, scipy.stats.rayleigh, scipy.stats.rice, scipy.stats.recipinvgauss, scipy.stats.semicircular,
        scipy.stats.skewnorm, scipy.stats.t, scipy.stats.trapezoid, scipy.stats.triang,scipy.stats.truncexpon,
        scipy.stats.truncnorm, scipy.stats.tukeylambda, scipy.stats.uniform, scipy.stats.vonmises, scipy.stats.vonmises_line, 
        scipy.stats.wald, scipy.stats.weibull_min,scipy.stats.weibull_max,scipy.stats.wrapcauchy]

    dist_all_names = np.array([str(dist_all[i].name) for i in range(len(dist_all))])  #names of all    
    dists = [dist_all[np.where(dist_all_names==distributions[i])[0][0]]for i in range(len(distributions))]  #get the distribution objects to test
    params = [tuple(parameters[i]) for i in range(len(parameters))]


    return dists, params

# -------------------------------------------------------------------------
# 5 - un-conditional and conditinal margins histograms
# -------------------------------------------------------------------------

def conditinal_margins_hist(F,data,names,condition_nodes):
    remaining_nodes = list_dif(list(range(len(names))),condition_nodes)
    nr_remaining_nodes = len(remaining_nodes)
    try:
        F_uncond = data.iloc[:,remaining_nodes].to_numpy()
        F_cond = np.array(F[0]).transpose()
        for i in range(nr_remaining_nodes):
            F_cond[:,i]
            plt.figure()
            plt.hist(F_uncond[:,i],bins=20,edgecolor = "black",color='silver',
                     label=['un-conditionalized\n mean: '+ str(round(np.mean(F_uncond[:,i]),1))])
            plt.hist(F_cond[:,i], bins=20,alpha=0.7,edgecolor = "black",color='cornflowerblue',
                     label=['conditionalized\n mean: '+ str(round(np.mean(F_cond[:,i]),1))]) 
            plt.legend()
            plt.ylabel("Count")
            plt.title(names[remaining_nodes[i]], fontsize=15)
    except:
        raise Exception('Check if argument Output in inference is equal to full')



