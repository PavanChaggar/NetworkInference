""" script containing class and functions for network loading and manipulation
"""
import numpy as np 

def adjacency_matrix(graph_path: str):
    if graph_path.endswith('.csv'):
        A = np.genfromtxt(graph_path, delimiter=',') 
    
    if graph_path.endswith('.graphml'):
        pass

    return A

def degree_matrix(adjacency_matrix): 
    pass
