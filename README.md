# Network Inference
This repository houses progress toward creating a flexible inference framework for network brain modelling of protein propagation. 

The current goals are to implement: 

- [ ] Simulations of protein propagation on brain networks using network diffusion models and Fisher-Kolmogorov–Petrovsky–Piskunov (FKPP) models. 
- [ ] Perform inference using sampling methods such as Markov chain Monte Carlo (MCMC), variational inference and simulation based inference. 
- [ ] Compare the efficacy and efficiency of these algorithms with variably sized parameter sets and network sizes. 



At present, the project will be implemented in python, making use of PyMC3 and SBI. In future, we hope to move to a Julia or C++ implementation, that are likely to scale more efficiently than python implementations. 