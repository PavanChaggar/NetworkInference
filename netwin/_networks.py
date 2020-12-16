""" script containing class and functions for network loading and manipulation
"""
import numpy as np 

def adjacency_matrix(graph_path):
    """Function to load graph from csv 
    args: 
        path : str
               string input pointing to the csv file containing the adjacency matrix of the graph
    returns:
           A : array/matrix
               square array containing the adjacency matrix 
    """
    if type(graph_path) == str:
        if graph_path.endswith('.csv'):
            A = np.genfromtxt(graph_path, delimiter=',') 
        elif graph_path.endswith('.graphml'):
            pass
    elif type(graph_path) == np.ndarray:
        A = graph_path

    return A

def degree_matrix(adjacency_matrix): 
    """Function to calculate degree matric from adjacency matric, A 
    args: 
        A : array/matrix
            square array containing the adjacency matrix 
    returns:
        D : array/matrix
            square array containing the degree matrix 
    """
    A = adjacency_matrix
    n = len(A) 
    D = np.diag([np.sum(A[i]) for i in range(len(A))]) 
    return D

def graph_Laplacian(A, D=np.array([])):
    """Function to generate the graph Laplacian from adjacency matrix
    args: 
        graph : array/matrix 
                square array containing the adjacency matrix 
    returns 
    L, Graph  : array/matrix
    Laplacian   square array containing the positive graph Laplacian given by L = D - A, where D is 
                degree matrix and A is the adjacency matrix
    """
 
    if D.any():
        L = D - A
    elif not D.any():
        D = degree_matrix(A) 
        L = D - A    
    return L
