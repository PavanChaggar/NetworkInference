# %% 
import numpy as np
import matplotlib.pyplot as plt
# %%
def f(theta):
    a, b = theta
    return a * np.exp(-b * t)
# %%
theta = [9, 2]
sigma = 1
n = 100
t = np.linspace(0, 2, n)

data = f(theta) + np.random.normal(0, sigma, n)
# %%
def make_log_normallikelihood(f, data, sigma, theta=None):
    def log_normallikelihood(theta):
        #if theta[-1] ** 0:
        #    theta[-1] += 1e-5
        return (-100 * np.sum((data - f(theta))**2))/sigma**2
    return log_normallikelihood

likelihood = make_log_normallikelihood(f=f, data=data, sigma=sigma)
# %%
def log_likelihood_ratio(theta, min, max, proposed_theta, log_likelihood):
    for p_n, parameter in enumerate(proposed_theta):
        if (parameter < min[p_n]) or (parameter > max[p_n]):
            return -np.inf

    return log_likelihood(proposed_theta) - log_likelihood(theta) 

def proposal(theta, cov):
    return np.random.multivariate_normal(theta,cov)

def accept_reject(r): 
    if r >= 1: 
        return True
    else: 
        x = np.random.uniform(0,1)
        if x < np.exp(r):
            return True
        else:
            return False

def mh(theta, min,max, samples, log_likelihood):
    cov = np.eye(len(theta)) * 0.01**2
    accepted_theta = np.empty((samples,len(theta)))
    n_accepted = 0 

    while n_accepted < samples:
        proposed_theta = proposal(theta, cov)
        r = log_likelihood_ratio(theta, min, max, proposed_theta, log_likelihood)
        if accept_reject(r):
            theta = proposed_theta
            accepted_theta[n_accepted, :] = theta
            n_accepted += 1
            print('accepted: %d' %n_accepted)
    return accepted_theta 

# %%
min = [1e-5, 1e-5]
max = [10,10]
result = mh(theta=[0,0], min=min, max=max, samples=20000, log_likelihood=likelihood)
plt.plot(result[2000:,0])# %%
plt.show()
plt.hist(result[2000:,0],100)
plt.show()
# %%
