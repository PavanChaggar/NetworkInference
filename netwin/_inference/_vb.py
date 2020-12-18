""" Scipt containing functions to implement variational inference using 
multivariate normal distribution and gamma distribution
"""

""" Scipt containing functions to implement variational inference using 
multivariate normal distribution and gamma distribution
"""

import numpy as np

from scipy.special import digamma 
from scipy.special import loggamma 

from scipy.stats import norm, gamma


def time_step(theta): 
    """Control time stepping interval proportional to the parameter magnitiude. 

    args: 
    theta:: float 
    """

    delta = theta * 1e-5
    if delta.any() < 0:
            delta = -delta
    if delta.any() < 1e-10:
            delta = 1e-10
    
    return delta

def central_difference(M, i, theta, delta, t):
    """ Calculate the derivate using a first order central difference approximation

    args:
        d : function 
            function on which to compute central difference 
        i : int 
            parameter index
    theta : np.array float 
            array of paramters for function, f
    delta : float 
            time-step for central difference
        t : np.array float
            time steps to evaluate at
    returns: 
        df : np.array, float 
             first order approximation, df, for time steps, t
    """
    dtheta = np.array(theta), np.array(theta)
    dtheta[0][i] += delta
    dtheta[1][i] -= delta
    f_1 = M.forward(u0=dtheta[0])
    f_2 = M.forward(u0=dtheta[1])
    den = (2 * delta)
    df = (f_1 - f_2) / den
    return df

def Jacobian(M, theta, t, n_params):
    """Compute the Jacobian for globally defined function, f, with parameter set Theta 
    
    args:
    theta : np.array, float
            parameters of global function, f
        t : np.array, float 
            time steps to evaluate at
    
    returns: 
        J : np.array, float
            Jacobian vector/matrix evaluated at theta, t. 
    """
    J = None
    #f_n = len(f)
    p_n = len(theta)

    for i in range(p_n):
        delta = time_step(theta[i])
        if J is None:
                J = np.zeros([len(theta[:-n_params]) * len(t), len(theta)], dtype=np.float32)
        df = central_difference(M, i, theta, delta, t)
        J[:,i] = df.flatten()
    
    return J

def free_energy(data, params, priors):

    N = len(data)

    m, p, c, s = params
    C = np.linalg.inv(p)

    m0,p0,c0,s0 = priors

    F = (N/2+c0-c)*(np.log(s)+digamma(c)) + s*c/2*(1/s-1/s0)
    F += c*np.log(s)+loggamma(c)
    _,logdetP = np.linalg.slogdet(p)
    F += logdetP/2
    F -= 1/2*(np.dot(np.dot((m-m0).transpose(),p0),m-m0)+np.trace(np.dot(C,p0)))
   
    return F

def parameter_update(error, params, priors, J):
    """ Update forward model function parameters theta in accordance with the update equations above
    
    args: 
        error : array, float
                vector of the difference between observations and model prediction 
       params : tuple
                parameters values
       priors : tuple 
                priors values
            J : array, float 
                array of Jacbian values from the model given parameter updates (calculated above)
    returns: 
       params : tuple
                updated parameters values
    """
    m, p, c, s = params
    m0, p0, _, _ = priors

    p_new = s*c*np.dot(J.transpose(), J) + p0
    c_new = np.linalg.inv(p_new)
    m_new = np.dot(c_new, (s * c * np.dot(J.transpose(), (error.flatten() +    np.dot(J, m))) + np.dot(p0, m0)))
    
    params[0][:], params[1][:] = m_new, p_new

    return params

def noise_update(error, data, params, priors, J):
    """ Update forward model function parameters phi in accordance with the update equations above
    
    args: 
        error : array, float
                vector of the difference between observations and model prediction 
       params : tuple
                parameters values
       priors : tuple 
                priors values
            J : array, float 
                array of Jacbian values from the model given parameter updates (calculated above)
    returns: 
       params : tuple
                updated parameters values
    """
    _, p, _, _ = params 
    _, _, c0, s0 = priors

    N = len(data)
    c = np.linalg.inv(p)
    c_new = N/2 + c0
    s_new = 1/(1/s0 + 1/2 * np.dot(error.flatten().transpose(), error.flatten()) + 1/2 * np.trace(np.dot(c, np.dot(J.transpose(), J))))
    
    params[2][:], params[3][:] = c_new, s_new

    return params

def error_update(y, M, theta, t):
    """Calculate difference between data and model with updated parameters

    args:
        y : array, float 
            vector of noisy data (observations)
    theta : array, float
            vector of parameter values for model, f
        t : array, float 
            vector of time steps at which to evaluate model 
    
    returns:
    error : array, float 
            vector of difference between noisy data and updated model
    """
    error = y - M.forward(u0=theta[0])
    
    return error

def vb(pm, M, data, t, params, priors, n_params, n): 
    m = np.zeros((n,len(params[0])))
    p = np.zeros((n, len(params[0]), len(params[0])))
    c = np.zeros((n))
    s = np.zeros((n))
    F = np.zeros((n))
    for i in range(n):
        #theta[i,:] = params[0]
        print('Iteration %d' %i)
        error = error_update(data, M, params, t)

        J = Jacobian(M, params[0], t, n_params)
        params = parameter_update(error, params, priors, J)
        m[i] = params[0]
        p[i] = params[1]
        params = noise_update(error, data, params, priors, J)
        c[i] = params[2]
        s[i] = params[3]
        F[i] = free_energy(data, params, priors)
    max_F = np.argmax(F)
    params = m[max_F], p[max_F], c[max_F], s[max_F]
    print('Finished!')
    return params, F

def fit(pm, n=20): 
    M = pm.model()
    data = pm.data()

    params = pm.m(), pm.p(), pm.c(), pm.s()
    priors = pm.m0(), pm.p0(), pm.c0(), pm.s0()

    m = np.zeros((n,len(params[0])))
    p = np.zeros((n, len(params[0]), len(params[0])))
    c = np.zeros((n))
    s = np.zeros((n))
    F = np.zeros((n))
    for i in range(n):
        #theta[i,:] = params[0]
        print('Iteration %d' %i)
        error = error_update(data, M, params, M.t)

        J = Jacobian(M, params[0], M.t, pm.n_params())
        params = parameter_update(error, params, priors, J)
        m[i] = params[0]
        p[i] = params[1]
        params = noise_update(error, data, params, priors, J)
        c[i] = params[2]
        s[i] = params[3]
        F[i] = free_energy(data, params, priors)
    max_F = np.argmax(F)
    params = m[max_F], p[max_F], c[max_F], s[max_F]
    print('Finished!')
    return params, F
