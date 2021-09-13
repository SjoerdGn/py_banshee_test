# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:51:04 2020
@author: Paul

Modified on Wed Jul 14 00:38:18 2021
@author: mmendozalugo

"""
import argparse 
import scipy.stats as st
import numpy as np
import pingouin as pg # normal, non-recursive, partial corr: pg.partial_corr
import matplotlib.pyplot as plt


def list_dif(li1, li2):
    return (list(list(set(li1)-set(li2)) ))

def bn_rankcorr(parent_cell, data, is_data,var_names,plot=0,fig_name=''):
    if is_data == 1:
        if data.shape[1] != len(parent_cell): 
            raise argparse.ArgumentTypeError('Number of data columns does not match the number of parent cells')  
        # Reading the number of variables    
        N = data.shape[1]
    
    if is_data == 0:
        if len(data) != len(parent_cell): 
            raise argparse.ArgumentTypeError('Number of data columns does not match the number of parent cells') 
        # Reading the number of variables    
        N = len(data)
    
    # Constructing a valid 'sampling order', which means that the node with no 
    # parents will be the first in the sampling order (SO) and so forth.
    SO = []
    while len(SO)<N:
        # Storing the elements in [0:N] not contained in [sampling_order]
        indices = list_dif(range(N),SO)
        for i in indices:
            # qq is empty if the parents of i are already contained in SO
            qq = list_dif(parent_cell[i],SO)
            if not qq:
                # in case qq is empty, adding i to the sampling order
                SO.append(i)
    # Creating an 'inverse order', so that:
    # SO(i) = j & invSO(j) = i ----> invSO(SO(i)) = i
    # Inverse order is used to fill in the symmetric R matrix
    invSO = SO[::-1]
    
    # Creating a data matrix out of specified rank correlation matrix, if such
    # input was chosen. 
    if is_data==0:
        import copy
        data_r_to_p = copy.deepcopy(data)
        for i in range(len(data)):
            for j in range(len(data[i])):
                # Transforming Spearman's into Pearson's correlation 
                # (auxiliary function 1)
                data_r_to_p[i][j] = ranktopearson(data_r_to_p[i][j])
 

    # Transforming the data to standard normal
    if is_data==1:
        norm_data,_ = NormalTransform(data)

    
    # Initializing the correlation matrix R
    R=np.zeros((N,N))
    np.fill_diagonal(R,1)
    
    # Initializing a cell containing informations during the recursive 
    # calculation of the correlation matrix given the network 
    # (see PartCorr function (3))
    L=[[None] * (N) for i in range(N)]
    
    # Starting the loop for recursively calculating the correlation matrix by
    # the second node
    for i in range(1,N):
        # Variables for the looping
        cond    = []            # Vector storing the conditionalized variables
        T       = np.zeros(i)   # Vector of the same length of previous nodes
        counter = 0             # Counter for recursive estimation
        
        seq  = parent_cell[SO[i]]      # Contains the parent of the i-th node
        seq2 = list_dif(SO[0:i], seq)  # Contains the previous (same order of 
                                       # SO!) nodes that are not parents 
                                       
        for j in seq:
            if is_data==0:
                #print('Yet to be implemented')
                T[counter] = data_r_to_p[SO[i]][len(cond)]
            else: 
                # Calculating the partial correlation between the node 
                # (normdata(:,SO(i))) at its parent (normdata(:,j)) given the
                # conditioning variable(s) (normdata(:,cond))
                T[counter] = pg.partial_corr(data=norm_data,
                                              x=norm_data.columns[SO[i]],
                                              y=norm_data.columns[j],
                                                covar=list(norm_data.columns[i]
                                                            for i in cond),
                                              method='pearson').r.values[0]                                              
            s=T[counter]

            for k in range(len(cond)-1,-1,-1):
                # Recursivelly calculating the correlation between nodes 
                # accounting for the conditional/partial correlation 
                # established by the BN (auxiliary function 3)
                [L,r1] = PartCorr(j, cond[k], cond[0:k],R,L,N+1) 
                # Based on the conditional/partial correlation, calculating the
                # resulting correlation coefficient (all the properties of the 
                # correlation matrix are guaranteed)
                shat   = s*np.sqrt((1-T[k]**2)*(1-(r1)**2))+T[k]*r1
                s      = shat
            if np.isnan(s):
                R[SO[i],j] = s 
                R[j,SO[i]] = s
                print('Error, s = nan')
                print('j= '+ str(j)+', k= '+str(k))
            else:
                # Saving the correlation coefficients calculated in the upper 
                # and lower triangle of the matrix R.
                R[SO[i],j] = s 
                R[j,SO[i]] = s
            counter += 1
            if not cond:
                cond = [j]
            else:
                cond.append(j)
        
        # Looping over the previous nodes (based on the ordering in SO) which 
        # are not parents, stored in seq2.
        for j in seq2:
            T[counter]=0
            s = T[counter]
            for k in range(len(cond)-1,-1,-1):
                if T[k]!=0 or s!=0:
                    # Recursively calculating the correlation between nodes 
                    # accounting for the conditional/partial correlation 
                    # established by the BN (auxiliary function 3)
                    L,r1 = PartCorr(j, cond[k], cond[0:k],R,L,N+1)
                    # Based on the conditional/partial correlation, calculating 
                    # the resulting correlation coefficient (all the properties 
                    # of the correlation matrix are guaranteed)
                    shat   = s*np.sqrt((1-T[k]**2)*(1-(r1)**2))+T[k]*r1
                    s      = shat
            
            # Storing the results
            R[SO[i],j] = s 
            R[j,SO[i]] = s
            counter    += 1
            if not cond:
                cond = [j]
            else:
                cond.append(j)
    # Transforming Pearson's correlation into Spearman's rank correlation 
    # (auxiliary function 2)
    R = pearsontorank(R)
    if plot==1:
        rank_corr_mat_fig(var_names,R)
            
    return R

# ----------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
# ----------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------
# 1 - Transforming Spearman's rank correlation into Pearson's correlation
# ----------------------------------------------------------------------------
def ranktopearson(R):
    # Transforming rank correlation (R) into Pearson's correlation (r)
    r = 2*np.sin((np.pi/6)*R)
    return r


# ----------------------------------------------------------------------------
# 2 - Transforming Pearson's correlation into Spearman's rank correlation
# ----------------------------------------------------------------------------
def pearsontorank(r):
	# Transforming Pearson's correlation (r) into rank correlation (R)
    R = (6/np.pi)*np.arcsin(r/2)
    return R

# ----------------------------------------------------------------------------
# 3 - Calculating the partial correlation with a recursive approach 
# (different from the standard Matlab function partialcorr).
# ----------------------------------------------------------------------------
def PartCorr(i , j, cond, R, L, m):
    # L: contains information about partial correlations
    # r: partial correlations
    corr_print='correlation between ' + str(i+1) +  ' and ' + str(j+1) + 'with cond ' + str([x+1 for x in cond])
    print(corr_print)
    # i,j:  nodes to calculate the correlation
    # cond: conditioning variable(s)
    # R:    correlation matrix (partially filled-in)
    # L:    a cell with information about estimates from previous steps 
    # m:    the number of nodes + 1

    # Ordering of the indeces
    s     = np.sort([i,j])
    isort = s[0]
    jsort = s[1]
    
    # Defining the number of conditioning variables
    n = len(cond)
    
    # If the conditioning variable vector is empty, then the value is
    # obtained from the correlation matrix    
    if n==0:
        r = R[i,j]
        return L,r
    
    # Extracting information on the cell L (from previous calculations)
    Lc = L[isort][jsort]
    
    # Calculating the index with auxiliary functions (4) and (6)
    if Lc:
       # If Lc has only one record, list has only one dimension, see except
       try:
           index = search_([item[0] for item in Lc],calc_val(cond,m))
       except IndexError as e:
           # print('Only one list in Lc')
           index = search_([Lc[0]],calc_val(cond,m))
       if index:
           r = Lc[index][1]
           return L,r
      
    [L,r1] = PartCorr(i,j,cond[1:n],R,L,m)
    [L,r2] = PartCorr(i,cond[0],cond[1:n],R,L,m)
    [L,r3] = PartCorr(j,cond[0],cond[1:n],R,L,m)
   
    # Calculating partial correlation of [(i,j) | (cond(1),cond(2:n))]
    r      = (r1-r2*r3)/((1-(r2)**2)*(1-(r3)**2))**(0.5)
    
    # Saving the results (auxiliary functions 4 and 5)
    L[isort][jsort] = add_corr(L[isort][jsort],calc_val(cond,m),r)
    
    return L,r

# ----------------------------------------------------------------------------
# 4 - Calculating an index for the recursive procedure in PartCorr
# ----------------------------------------------------------------------------
def calc_val(cond,m):
    # v: value of the index
    # cond: conditioning variable(s) from auxiliary function 3
    # m:    the number of nodes + 1

    n  = len(cond)
    sc = np.sort(cond)
    if(n==0):
        v = 0
        return v

    v = 0
    for i in range(0,n):
        v = (v + (sc[i]+1)*m**i)
    return v

# ----------------------------------------------------------------------------
# 5 - Adding correlation to cell L
# ----------------------------------------------------------------------------
def add_corr(Lc,val,r):
    # Lc: correlation to be added
    # val: value of the index from auxiliary function 4
    # r:   partial correlation from auxiliary function 3
    
    if not (Lc):
        Lc = [val, r]
        return Lc
    # If Lc has only one record, list has only one dimension, e.g. [2,0.378]
    try:
        if np.issubdtype(Lc[0], np.integer):
            Lc1=[Lc[0]]
        # Else: get all first elements from a list like [[0,0,257],[2,0.378]]
        else:
            Lc1 = [item[0] for item in Lc]
    except TypeError as e:
        Lc1 = [item[0] for item in Lc]

    n1 = 0
    n2 = len(Lc1)+1
    
    while( (n2-n1) >= 2 ):
        n = int(np.ceil((n1+n2)/2))
        if(Lc1[n-1] == val):
            index = n-1
            return Lc
        else:
            if(Lc1[n-1]<val):
                n1=n
            else:
                n2=n
    

    # If Lc has only one record, list has only one dimension, e.g. [2,0.378]
    try:
        if np.issubdtype(Lc[0], np.integer):
            Lc=[Lc]
    except TypeError as e:
        # print('more than one elements')
        pass        
    
    Lc.insert(n1,[val,r])
    # Lc.append([val,r])
    return Lc


# ----------------------------------------------------------------------------
# 6 - A search function for the recursive procedure in PartCorr
# ----------------------------------------------------------------------------
def search_(Lc,val):
    # index: an index for the recursive procedure
    # Lc:  value of L from previous calculations
    # val: value of the index from auxiliary function 4
    n1 = 0
    n2 = len(Lc)-1
    while n1<=n2:
        n = np.floor((n1+n2)/2).astype('int')
        if Lc[n] == val:
            index = n
            return index
        else:
            if Lc[n]<val:
                n1 = n+1
            else:
                n2 = n-1
    index = 0
    return index

# ----------------------------------------------------------------------------
# 7 - Transforming data into ranked (uniform) and standard normal distribution
# ----------------------------------------------------------------------------
def NormalTransform(data): 
    M    = data.shape[0]         # Reading number of observations per node
    ranks = data.rank(axis=0)
    u_hat     = ranks/(M+1)
    norm_data = u_hat.apply(np.vectorize(st.norm.ppf))
    norm_data.replace([np.inf, -np.inf], 0, inplace=True) # Adjusting abnormal values
    return norm_data,u_hat
# ----------------------------------------------------------------------------
# 8 - Plot Rank Correlation matrix
# ----------------------------------------------------------------------------
  
def rank_corr_mat_fig(NamesBN,RBN):
    '''
    Parameters
    ----------
    NamesBN : list
        BN nodes names from model.
    RBN : numpy.ndarray
        Correlatuon matrix.

    Returns
    -------
    None.

    '''  
    nam =NamesBN
      
    #Replace 0 with NaN
    z= np.round(RBN,4)
    z = z.astype('float')
    z[z == 0] = 'nan' # or use np.nan

    #reduce the names of the variables if the number of variables is large 
    #to plot the only the reduced names 
    if len(nam)>80:
        nv = int(round(len(nam)/26,0))
    elif len(nam)<=80 and len(nam)>40:
        nv = int(round(len(nam)/13,0))
    elif len(nam)<=40 and len(nam)>20:
        nv = int(round(len(nam)/21,0))
    elif len(nam)<=20:
        nv = len(nam)
    
    px = list(range(len(nam)))  #position of the labels
    
    fig, ax = plt.subplots(figsize=(13,9))
    im = plt.imshow(z,cmap='Blues', interpolation='nearest')
    plt.colorbar()
    
    if len(nam)<=20:
        plt.xticks(px,nam,rotation=45)
        plt.yticks(px,nam)
    else: #plot the only the reduces names so the plot doesnt look saturated
        plt.xticks(px[::nv],nam[::nv],rotation=45)
        plt.yticks(px[::nv],nam[::nv])
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    

    if len(nam)<=10: 
        zz= np.round(RBN,4)
        zz = zz.astype(str)
        for i in range(len(nam)):
            for j in range(len(nam)):
                if zz[i,j]=='0.0':
                    zz[i,j] = ''
                text = ax.text(j, i, zz[i, j],
                               ha="center", va="center", 
                                color="k",fontsize='small',fontweight='roman')   
    plt.show()

    return None
