#%% 
from netwin import *
import os, fnmatch
import matplotlib.pyplot as plt
from nilearn import plotting
import numpy as np
import pandas as pd
import matplotlib as mpl

%load_ext autoreload
%autoreload 2
# %%
result =[]
path = '/home/chaggar/Documents/Connectomes/standard_connectome/scale1/subjects'
for root, dirs, files in os.walk(path):
    for name in files:
        if fnmatch.fnmatch(name, 'fdt_network_matrix'):
            result.append(os.path.join(root, name))

# %%
A_all = np.ones((83,83,10))
for i, j in enumerate(result):
    A_all[:,:,i] = np.genfromtxt(j)
# %%
A = np.mean(A_all,axis=2)

A = A / np.max(A)

SymA = (A + A.T)/np.max(A+A.T)
# %%
m = NetworkFKPP(A)

p = np.zeros([83]) + 1e-5
mask = [25, 26, 39, 40, 66, 67, 80, 81]
p[mask] = 0.1

k = 5
a = 10

m.t = np.linspace(0,1,100)

u0 = np.append(np.log(p), np.log([k, a]))

sim = m.forward(u0)
# %%
coordinate_path = '/home/chaggar/Documents/Network_Inference/data/mni_coordinates/mni-parcellation-scale1_coordinates.csv'

nodes = pd.read_csv(coordinate_path)

x, y, z = np.array(nodes.x), np.array(nodes.y), np.array(nodes.z)
coords = np.vstack((x, y, z))

# %%
alpha = np.linspace(0.7, 1, 258, endpoint=True)
degree_norm = np.diagonal(m.D()) / np.max(m.D())
node_colour = plt.cm.Blues(degree_norm, alpha)

# %%
plotting.plot_connectome(node_coords=coords.T,adjacency_matrix=SymA, node_size = 30, colorbar=True, node_color=node_colour, alpha=0.5, edge_cmap='Reds', edge_vmin=0, edge_vmax=1)
plotting.show()
# %%
mpl.rc_file_defaults()
plotting.plot_matrix(A)
plotting.show()

# %%
alpha = np.linspace(0, 1, 258, endpoint=True)
p_conc = sim[50]/np.max(sim[50])
node_colour = plt.cm.Greens(p_conc, alpha)
# %%
plotting.plot_connectome(node_coords=coords.T,adjacency_matrix=np.eye(83), node_size = 30, node_color=node_colour)
plotting.show()
# %%
'/home/chaggar/Documents/plots'
mpl.style.use('ggplot')
plt.plot(m.t, sim, c='g', alpha=0.1, linewidth=2)
plt.ylabel("Protein Concentration")
plt.xlabel("Time")
plt.savefig('/home/chaggar/Documents/PLOTS/Simulated_time.png', dpi=1000)
plt.show()
# %%
data = np.empty_like(sim)
for i in range(len(sim[0])):
    data[:,i] = sim[:,i] + (np.random.rand(len(sim[:,i]))/10)
# %%
mpl.style.use('ggplot')
plt.plot(m.t, data, c='b', alpha=0.1, linewidth=1)
plt.ylabel("Protein Concentration")
plt.xlabel("Time")
plt.savefig('/home/chaggar/Documents/PLOTS/Noise_time.png', dpi=1000)
plt.show()

# %%
# set priors 
data = np.empty_like(sim) 
for i in range(len(sim[0])):
    data[:,i] = sim[:,i] + (np.random.rand(len(sim[:,i]))/10)

p0 = np.zeros([83])
k0 = 0
a0 = 0
u_0 = np.append(p0, [k0, a0])
n=80

problem = VBProblem(model=m, data=data, init_means=u_0)

sol, F = problem.infer(n=n)

plt.plot(range(n), F, alpha=0.8, label='Free Energy')
plt.show()
# %%
mpl.style.use('ggplot')
plt.bar(range(83), data[0], alpha=0.5, color='b', label='data')
plt.bar(range(83), sim[0], alpha=1, color='g', label='simulated')
plt.bar(range(83), np.exp(sol[0][:-2]), alpha=0.5, color='r', label='inferred')
plt.xlabel("Node ID")
plt.ylabel("Initial Protein Concentration")
plt.legend()
#plt.savefig('/home/chaggar/Documents/PLOTS/initial_conc.png', dpi=1000)
plt.show()

# %%
inferred = m.forward(sol[0])
inf = plt.plot(m.t, inferred, c='r', alpha=0.3, linewidth=1, label='inferred')
si = plt.plot(m.t, sim, c='g', alpha=0.2, linewidth=1, label='sim')
plt.ylabel("Protein Concentration")
plt.xlabel("Time")
plt.legend((inf[0], si[0]), ('inferred', 'simulated'))
#l2 = plt.legend(si, ['simulated'])
#plt.gca().add_artist(l1)
#plt.savefig('/home/chaggar/Documents/PLOTS/inferred_time.png', dpi=1000)
#plt.show()
# %%
plt.plot(range(n), F, alpha=0.8, label='Free Energy')
plt.ylabel("Free Energy")
plt.xlabel("Iteration #")
plt.legend()
#plt.savefig('/home/chaggar/Documents/PLOTS/free_energy.png', dpi=1000)
plt.show()

#divide elemnts by s.d of column diagonals 


# %%
import numpy as np
from scipy.special import digamma 
from scipy.special import loggamma 

from scipy.stats import norm, gamma
import matplotlib.pyplot as plt

def plot_posterior(means,cov,labels=None,samples=None,actual=None):
    """
    helper function for plotting posterior distribution
    
    Parameters
    ----------
    means : array like
    cov   : matrix  
    labels : list 
    samples : 2D (samples x params)
              as ouput by MH
    actual : array like
             true parameter values if known

    Returns
    -------
    matplotlib figure
    
    """
    fig = plt.figure(figsize=(10,10))
    n     = means.size
    nbins = 50
    k     = 1
    for i in range(n):
        for j in range(n):
            if i==j:
                x = np.linspace(means[i]-5*np.sqrt(cov[i,i]),
                                means[i]+5*np.sqrt(cov[i,i]),nbins)
                y = norm.pdf(x,means[i],np.sqrt(cov[i,i]))

                plt.subplot(n,n,k)            
                plt.plot(x,y)
                if samples is not None:
                    plt.hist(samples[:,i],histtype='step',density=True)
                if labels is not None:
                    plt.title(labels[i])
                if actual is not None:
                    plt.axvline(x=actual[i],c='r')
                    
            else:
                m = np.asarray([means[i],means[j]])
                v = np.asarray([[cov[i,i],cov[i,j]],[cov[j,i],cov[j,j]]])
                xi = np.linspace(means[i]-5*np.sqrt(cov[i,i]),
                                 means[i]+5*np.sqrt(cov[i,i]),nbins)
                xj = np.linspace(means[j]-5*np.sqrt(cov[j,j]),
                                 means[j]+5*np.sqrt(cov[j,j]),nbins)
                x  = np.asarray([ (a,b) for a in xi for b in xj])
                x  = x-m
                h = np.sum(-.5*(x*(x@np.linalg.inv(v).T)),axis=1)

                h = np.exp(h - h.max())
                h = np.reshape(h,(nbins,nbins))
                plt.subplot(n,n,k)        

                plt.contour(xi,xj,h)
                
                if samples is not None:
                    plt.plot(samples[:,i],samples[:,j],'k.',alpha=.1)
            k=k+1

    plt.show()
    return fig
# %%
plot_posterior(sol[0][-2:], sol[1][-2:,-2:], actual=np.log([5.0,10.0]))

# %%
cov = np.linalg.inv(sol[1])
diag = np.diag(cov)
mat = np.sqrt(np.outer(diag,diag.T))
cor = cov/mat

plt.imshow(cor, cmap='RdBu_r', vmin=-1.0,vmax=1.0)
plt.colorbar()
plt.savefig('/home/chaggar/Documents/PLOTS/corr_matrix.png', dpi=1000)
plt.show()

# %%
