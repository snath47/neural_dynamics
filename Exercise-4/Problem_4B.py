import numpy as np
import matplotlib.pyplot as plt
import unicodeit
import os

class ML_nullcline_plotter:
    def __init__(self, name) -> None:
        self.param = {
        "V1" : -1.2,
        "V2" : 18.0,
        "V3" : 2.0,
        "V4" : 30.0,
        "g_K": 8.0,
        "g_L": 2.0,
        "V_Ca": 120.0,
        "V_K": -84.0,
        "V_L": -60.0,
        "C": 20.0,
        "phi": 0.041,
        }
        self.name = name
        
    def activation_variables(self, V):
        M_ss = (np.tanh((V - self.param["V1"])/self.param["V2"])+1.0)/2.0
        W_ss = (np.tanh((V - self.param["V3"])/self.param["V4"])+1.0)/2.0
        tau_w = 1.0/(self.param["phi"] * np.cosh((V-self.param["V3"])/(2*self.param["V4"])))
        return M_ss, W_ss, tau_w
    
    def plotter(self, I_ext, g_Ca, num_fp):
        plt.clf()
        V = np.linspace(-200,300,500)
        M_ss, W_ss, _ = self.activation_variables(V)
        W1 = (I_ext - g_Ca * np.multiply(M_ss,(V - self.param["V_Ca"])) - self.param["g_L"] * (V - self.param["V_L"]))/(self.param["g_K"]*(V - self.param["V_K"]))
        W2 = W_ss
        plt.plot(V, W1, label = 'v-nullcline')
        plt.plot(V, W2, label = 'w-nullcline')
        plt.legend(loc = 'upper left')
        plt.xlabel('V (in mV)')
        plt.ylabel('W (in mV)')
        plt.title("Nullcline plot with "+ unicodeit.replace('$I_\{ext\}$')+" = " +str(I_ext)+ " and " + unicodeit.replace('$g_\{Ca\}$') +" = "+ str(g_Ca))
        plt.savefig("graphs/"+self.name+ "_"+ str(num_fp) + '.png')

    def optimal_param_finder(self):
        V = np.linspace(-200,200,200)
        M_ss, W_ss, _ = self.activation_variables(V)
        Iext_range=np.linspace(0, 100,101)
        Iext1 = None
        Iext2 = None
        Iext3 = None
        g_Ca1 = None
        g_Ca2 = None
        g_Ca3 = None
        g_Ca_range = np.flip(np.linspace(-2200, 20000, 5000))
        #g_Ca_range = np.concatenate((np.linspace(4.4, 2000.0, 2500), np.linspace(-2500.0, 4.4, 2500)))
        for g_Ca in g_Ca_range:    
            for Iext in Iext_range:    
                if Iext1 is not None and Iext2 is not None and Iext3 is not None:
                    break
                W1 = (Iext - g_Ca * np.multiply(M_ss,(V - self.param["V_Ca"])) - self.param["g_L"] * (V - self.param["V_L"]))/(self.param["g_K"]*(V - self.param["V_K"]))
                W2 = W_ss
                idx = np.argwhere(np.diff(np.sign(W1 - W2))).flatten()    
                if Iext1 is None and len(idx) == 1:
                    Iext1 = Iext
                    g_Ca1 = g_Ca
                elif Iext2 is None and len(idx) == 2:
                    Iext2 = Iext
                    g_Ca2 = g_Ca
                elif Iext3 is None and len(idx) == 3:
                    Iext3 = Iext
                    g_Ca3 = g_Ca
        return (Iext1,g_Ca1), (Iext2,g_Ca2), (Iext3,g_Ca3)
        

if __name__ == "__main__":
    path = "graphs"
    if not os.path.exists(path):
        os.makedirs(path)
    new_plot = ML_nullcline_plotter("ML_nullcline_4B")
    graph_param_list = new_plot.optimal_param_finder()
    print(graph_param_list)
    for i in range(len(graph_param_list)):
        new_plot.plotter(graph_param_list[i][0], graph_param_list[i][1], i+1)