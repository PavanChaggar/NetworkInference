""" script containing class and functions for modelling, integrating etc
""" 

def model(model: str): 
    if model == 'diffusion': 
        return network_diffusion

def network_diffusion(u0, params, t):
    p = u0
    L, k = params
    du = k * (-L @ p)
    return du

def solve(model, u0, params, t):
    return odeint(model, u0, t, args=(params,))
