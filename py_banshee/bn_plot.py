# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:41:40 2021
@author: Paul

Modified on Wed Jul 14 00:38:18 2021
@author: mmendozalugo
"""

import networkx as nx # import pygraphviz
import graphviz as gv
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import Image
import os
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"   #to use dot graphviz


def bn_visualize(parent_cell, R, names, data=None,fig_name=''):
    """ Visualize the structure of a defined Bayesian Network
 
     bn_visualize creates a directed digraph presenting the
     structure of nodes and arcs of the Bayesian Network (BN), defined by
     parent_cell. The function also displays the conditional rank
     correlations at each arc defined by R.
 
     INPUT. The required input is as follows:
 
        parent_cell     A cell array containing the structure of the BN,
                        the same as required in the bn_rankcorr function
        R               A matrix generated using bn_rankcorr function
        names           A list with the names of the columns/rows. Should
                        be in the same order as they appear in matrix R and
                        parent_cell
        fig_name        Name of the .png file with the Bayesian Network that
                        is created. The file is saved in the working directory
                        by default. Add a full folder path to save elsewhere
 
     OPTIONS. bn_visualize(DATA,R,NAMES) has the following option:
 
        NAMES           A cell array containing names of the nodes of the
                        BN; otherwise, default names from data are assigned.
 
    """
    
    G = nx.DiGraph()
    if isinstance(data, pd.DataFrame):
     for node in data:
        plt.figure()
        h=sns.histplot(data[node],kde=True)
        #h=sns.distplot(data[node],kde=True) #if error in histplot
        h.set_xlabel('')
        h.set_title('{}'.format(node),fontsize=25)
        plt.savefig('histogram_{}.png'.format(node))
        G.add_node(node,image='histogram_{}.png'.format(node),
                   fontsize=0)
        plt.show()
    else:
        G.add_nodes_from(names,style='filled',fillcolor='red')
        plt.show()
        
        
    for i in range(len(names)):
        parents=parent_cell[i]
        for j in parents:
            G.add_edge(names[j],names[i],label=("%.2f")%R[j,i],
                       fontsize=18)
            
    nx.drawing.nx_pydot.write_dot(G, 'BN_visualize_{}'.format(fig_name))
    # Convert dot file to png file
    gv.render('dot', 'png', 'BN_visualize_{}'.format(fig_name))
    
    def deleteFile(filename):
        if os.path.exists(filename) and not os.path.isdir(filename) and not os.path.islink(filename):
            os.remove(filename)
            
    deleteFile('BN_visualize_{}'.format(fig_name))  
	
    return Image(filename= 'BN_visualize_{}'.format(fig_name)+'.png')
