""" script containing class and functions for modelling, integrating etc
""" 
import netwin as nw

class Model(object): 
    
    def __init__(self, network_path: str, model_name: str):
        
        self.filename = network_path
        self.which_model = model_name

        self.A = nw.adjacency_matrix(self.filename)
        self.D = nw.degree_matrix(self.A)
        self.L = nw.graph_Laplacian(A=self.A, D=self.D)

        self.f, self.which_model = self._models(model_choice = self.which_model)

    def _models(self, model_choice: str): 
        #print(model_choice)
        if model_choice == 'network_diffusion': 
            f = nw.network_diffusion
            return f, model_choice
        elif model_choice == 'fkpp':
            pass 

    #def _network_diffusion(self, u0, t, params):
    #    p = u0
    #    L, k = params
    #    du = k * (-L @ p)
    #    return du

    def solve(self, model, u0, t, params):
        return odeint(self.f, u0, t, args=(params,))
