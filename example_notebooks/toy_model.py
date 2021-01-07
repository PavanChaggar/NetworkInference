# %%
import netwin as nw 
from netwin.models import NetworkFKPP
from netwin import infer, VBModel

import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(10)

# %%
A = np.reshape(np.random.normal(5, 2, 25), (5, 5))
np.fill_diagonal(A, 0)

A = A / np.max(A)

m = NetworkFKPP(A)

p = np.random.uniform(0,0.5,5)

u0 = np.append(p, [2.0, 3.0])

m.t = np.linspace(0,1,50)

sim = m.forward(np.log(u0))

plt.plot(m.t, sim)
plt.show()
# %%
p0 = np.ones([5])
u_guess = np.append(p0, [1.0,1.0])

ProbModel = VBModel(model=m, data=sim, init_means=u_guess)

pm = infer(ProbModel, n=20)
# %%
mean_a = np.exp(pm.m()[-2:])
cov_a = pm.Cov()[-2:,-2:]

samples = np.random.multivariate_normal(mean_a, cov_a, 100000)

(counts, x_bins, y_bins) = np.histogram2d(samples[:, 0], samples[:, 1])
plt.contourf(counts, extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]])
plt.show()
# %%
