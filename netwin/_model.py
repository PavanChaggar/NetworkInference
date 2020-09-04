""" script containing class and functions for modelling, integrating etc
""" 
import netwin as nw

class model: 
    
    def __init__(self,
                 network_path: str,
                 model_name: str)
        
        self.filename = network_path
        self.which_model = model_name

        self.A = nw.adjacency_matrix(self.filename)
        self.D = nw.degree_matrix(self.A)
        self.L = nw.graph_Laplacian(A=self.A, D=self.D)
    
        self.f, self.which_model = _model(model_name = self.which_model)

    def _model(model_name: str): 
        if model == 'diffusion': 
            return _network_diffusion, model_name
        if model == 'fkpp':
            pass 

    def _network_diffusion(u0, t, params):
        p = u0
        L, k = params
        du = k * (-L @ p)
        return du

    def solve(model, u0, t, params):
        return odeint(model, u0, t, args=(params,))
