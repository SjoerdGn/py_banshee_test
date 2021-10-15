# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 11:46:13 2021

@author: mmendozalugo
"""
from py_banshee.rankcorr import bn_rankcorr
from py_banshee.bn_plot import bn_visualize
from py_banshee.copula_test import cvm_statistic
from py_banshee.d_cal import gaussian_distance
from py_banshee.prediction import inference,conditional_margins_hist

import numpy as np
import pandas as pd

#%% Data 
np.random.seed(123)
# Define location of data file
data = pd.read_csv('cities.csv')

# Define name of output figure with BN; used by bn_visualize
fig_name = 'bn_cities'

# Select the columns to use in the NPBN
columns_used=[0, 6, 7, 8, 3] # climate, arts, recreation, economics, 
                             # safety
data = data.iloc[:,columns_used] 
#%%Defining the structure of the BN
names = list(data.columns)
# Extract number of nodes from data
N = data.shape[1]

#Structure of the BN
parent_cell = [None]*N
parent_cell[0] = []           # climate (no parents)
parent_cell[1] = [2]          # arts (parent node: recreation)
parent_cell[2] = [3, 0]       # recreation (parent nodes: economics, climate)
parent_cell[3] = []           # economics (no parents)
parent_cell[4] = [1, 2, 3, 0] # safety (parents: all other variables)

#%% bn_rankcorr - Conditional rank correlation matrix
R=bn_rankcorr(parent_cell,data,is_data = 1,plot=1,var_names=names)

# The plot shows that safety has the strongest correlation with arts,
# then the conditional correlation with recreation. The conditional 
# correlations with with economics and climate are weaker. Between other 
# variables, there is strong dependency between arts and recreation.

#%% bn_visualize - Plot of the Bayesian Network
bn_visualize(parent_cell,R,data.columns,data=data,fig_name=fig_name+'_margins')
# The plot presents the BN with 5 nodes and 7 arcs, with the (conditional)
# rank correlations indicated on the arcs.

#%% CVM_STATISTICS - test goodness-of-fit of the Gaussian copula 
M = cvm_statistic(data, plot=1, names=data.columns,fig_name=fig_name)

#%% D-cal
SampleSize_ERC_NRC = 4000
SampleSize_NRC_BNRC = 400
no_iterations = 500

D_ERC,B_ERC,D_BNRC,B_BNRC = gaussian_distance(R,data,
                                              SampleSize_ERC_NRC,
                                              SampleSize_NRC_BNRC,
                                              no_iterations,1,'H',fig_name)

#%% Inference

condition_nodes = [0,1,2,3] #conditionalized variables, all except for safety (predict)
condition_values = data.iloc[:,condition_nodes].to_numpy()

# Due to coding in inference, only works for n>1 (2nd argument)
F = inference(condition_nodes,
              condition_values,
              R,
              data,
              SampleSize=329,
              empirical_data=1, 
              Output='full')
#%% un-conditional and conditional marginal histograms
#  only works if argument Output = 'full' in inference 
conditional_margins_hist(F,1,data,names,condition_nodes)
