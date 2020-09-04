""" Scipt containing different network models 
"""
def network_diffusion(u0, t, params):
    p = u0
    L, k = params
    du = k * (-L @ p)
    return du