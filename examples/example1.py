# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 13:45:32 2021

@author: mmendozalugo
"""

from py_banshee.rankcorr import bn_rankcorr
from py_banshee.bn_plot import bn_visualize
from py_banshee.prediction import inference

#Defining the variables of the BN
names = ['V1','V2','V3']  #names of the variables (nodes)
N = len(names) 		      #number of nodes

#parametric distributions of the nodes
distributions = ['norm','genextreme','norm']	
parameters = [[100,23],[-0.15,130,50],[500,100]]

#Defining the structure of the BN
ParentCell = [None]*N
ParentCell[0] = []
ParentCell[1] = []
ParentCell[2] = [0,1]

#Defining the rank correlation matrix
RankCorr = [None]*N
RankCorr[0] = []
RankCorr[1] = []
RankCorr[2] = [.41,-.25]

#Conditional rank correlation matrix
R = bn_rankcorr(ParentCell,RankCorr,is_data = 0,var_names=names,plot=1)

#Plot of the Bayesian Network
bn_visualize(ParentCell,R,names,fig_name='BN_TEST')

# Inference
condition_nodes = [0] #conditionalized variables (node V1)
condition_values = [181] #conditionalized value (node V1)

F = inference(Nodes = condition_nodes,
              Values = condition_values,
              R=R,
              DATA=[],
              SampleSize=1000,
              empirical_data=0, 
              distributions=distributions,
              parameters=parameters,
              Output='full')
