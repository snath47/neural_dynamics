import networkx as nx
import numpy as np
from sdeint import itoEuler

stored_v = [None] * 30

class FHNNeuralNetwork:
    def __init__(self, n = 30, p = 0.7, t = 1000, T_e = 5) -> None:
        self.t = t
        self.n = n
        self.T_e = T_e
        G = nx.generators.random_graphs.gnp_random_graph(self.n, p)
        nx.draw(G)
        self.A = nx.adjacency_matrix(G).toarray()
        
    def initialize_network(self):
        Network_List = []
        for i in range(self.n):
            Network_List.append(FHNNeuron(0.05, 0.756, 0.001, 0.25, 0.005, self.A, self.T_e, i))
        
        

class FHNNeuron:
    def __init__(self, epsilon, beta, I, G_E, sigma, A, T_e, i) -> None:
        self.epsilon = epsilon
        self.beta = beta
        self.I = I
        self.G_E = G_E
        self.sigma = sigma
        self.A = A
        self.t_e = T_e
        self.i = i
        
    def Esyn(self, v, t):
        Es = 0
        for j in range(len(stored_v)):
            if self.i != j:
                Es += self.A[self.i, j] * self.G_E * (stored_v[0][j] - v)
    
    def F(self, U, t):
        v, w = U
        dvdt = v - (v**3)/3 - w + self.I + self.Esyn(v, t)
        dwdt = self.epsilon(v + self.alpha - self.beta*w)
        stored_v[-1][self.i] = v + dvdt
        return np.array([dvdt, dwdt])
    
    def G(self, U, t):
        return np.diag([self.sigma, 0.0])
            
        
    def simulate(self, time, U_0):
        self.solution = itoEuler(self.F, self.G, U_0, time)
