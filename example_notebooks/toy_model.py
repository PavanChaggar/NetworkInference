import netwin as nw 
import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(10)

A = np.reshape(np.random.normal(5, 2, 25), (5, 5))
np.fill_diagonal(A, 0)

m = nw.NetworkFKPP(A)

u0 = np.array([0.0,0.0,1.0,0.0,1.0,5.0,10.0])
m.t = np.linspace(0,1,100)
sim = m.forward(u0)

plt.plot(m.t, sim)
plt.show()