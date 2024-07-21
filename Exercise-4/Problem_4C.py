import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

class ML_plotter:
    def __init__(self, Iext) -> None:
        self.Iext = Iext
        self.param = {
        "V1" : -1.2,
        "V2" : 18.0,
        "V3" : 2.0,
        "V4" : 30.0,
        "g_Ca": 4.4,
        "g_K": 8.0,
        "g_L": 2.0,
        "V_Ca": 120.0,
        "V_K": -84.0,
        "V_L": -60.0,
        "C": 20.0,
        "phi": 0.041,
        }
    
    def activation_variables(self, V):
        M_ss = (np.tanh((V - self.param["V1"])/self.param["V2"])+1.0)/2.0
        W_ss = (np.tanh((V - self.param["V3"])/self.param["V4"])+1.0)/2.0
        tau_w = 1.0/(self.param["phi"] * np.cosh((V-self.param["V3"])/(2.0*self.param["V4"])))
        return M_ss, W_ss, tau_w

    def dUdt(self, U, t):
        """
        Defines the differential equations for the ML model for neuron.

        Arguments:
            U :  vector of the state variables:
                    U = [V, W]
            t :  time
            Iext: external current as parameter
            
        Returns:
            an array containing the differentials of the state variables
        """
        
        V, W = U
        M_ss, W_ss, tau_w = self.activation_variables(V)
        dVdt =  self.Iext - self.param["g_Ca"] * M_ss * (V-self.param["V_Ca"]) - self.param["g_K"] * W * (V-self.param["V_K"]) - self.param["g_L"] * (V-self.param["V_L"])
        dWdt = (W_ss - W)/tau_w
        return [dVdt, dWdt]
    
    def time_series_solver(self, t):
        U_0 = [0., 0.]
        return odeint(self.dUdt, U_0, t)
    
    def phase_plot_solver(self):
        v = np.linspace(-80., 80., 35)
        w = np.linspace(0., .6, 35)
        
        V, W = np.meshgrid(v, w)
        dV = np.zeros(V.shape)
        dW = np.zeros(W.shape)
        
        shape1, shape2 = W.shape
        
        for indexShape1 in range(shape1):
            for indexShape2 in range(shape2):
                dUdtAtV = self.dUdt([V[indexShape1, indexShape2], W[indexShape1, indexShape2]], 0)
                dV[indexShape1, indexShape2] = dUdtAtV[0]
                dW[indexShape1, indexShape2] = dUdtAtV[1]
                
        return V, W, dV, dW
    
if __name__ == "__main__":
    path = "graphs"
    if not os.path.exists(path):
        os.makedirs(path)
    t = np.linspace(0, 1000, 3000)
    Iext_list = [40, 90]
    for i in range(len(Iext_list)):
        plot = ML_plotter(Iext_list[i])
        solution = plot.time_series_solver(t)
        #time series
        fig, (ax1, ax2) = plt.subplots(2)
        plt.suptitle('Time Series', fontsize = 14)
        ax1.plot(t, solution[:, 0])
        ax1.set_title('V vs t')
        ax1.set_ylabel("Membrane potential (in mV)")
        ax2.set_title('W vs t')
        ax2.plot(t, solution[:, 1], 'tab:red')
        ax2.set_xlabel("Time (in msec)")
        ax2.set_ylabel("Recovery Variable (in mV)")
        fig.subplots_adjust(hspace=.5)
        fig.savefig("graphs/"+"Time Series_4C_"+str(i+1))
        #plot phase potrait and state space trajectory
        plt.clf()
        plt.figure(figsize=(8,8))
        V, W, dV, dW = plot.phase_plot_solver()
        plt.quiver(V, W, dV, dW, color = 'black',width = 0.0040)
        plt.xlim(-80, 80)
        plt.ylim(0., .6)
        plt.plot(solution[:,0], solution[:,1], color = 'r')
        plt.title('Phase Potrait', fontsize = 14)
        plt.xlabel('V (in mV)')
        plt.ylabel('W (in mV)')
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.savefig('graphs/'+'Phase_Plot_4C_'+str(i+1))
