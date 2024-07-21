import numpy as np
from sdeint import itoEuler
import matplotlib.pyplot as plt
import math
import os
from scipy.signal import find_peaks
from tqdm import tqdm
import unicodeit
import glob

class stochastic_HH:
    def __init__(self, Iext, sigma, duration = 200.0, question_no = None) -> None:
        self.Iext = Iext
        self.sigma = sigma
        self.question_no = question_no
        self.duration = duration
        
    def gating_variables(self, V):
        """
        Generates the gating variables for a given V

        Args:
            V (int): neuron membrane potential

        Returns:
            int array: gating variables
        """
        e = math.e
        return [
            (25 - V)/(10*(e**((25-V)/10) - 1)),
            4 * (e ** (-V/18)),
            (7/100)*(e ** (-V/20)),
            1/(1 + e ** ((30 - V)/10)),
            (10 - V)/(100*(e**((10-V)/10) - 1)),
            (1/8)* (e ** (-V/80))    
    ]
        
    def dUdt(self, U, t):
        """
        Defines the differential equations for the HH model for neuron.

        Arguments:
            U :  list of the state variables:
                    U = [V,n,m,h]
            t :  time
            Iext: external current as parameter
            
        Returns:
            an array containing the differentials of the state variables
        """
        V, n, m, h = U
        alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = self.gating_variables(V)
        Cm = 1
        g_k = 36
        g_Na = 120
        g_L = 0.3
        V_k = -12
        V_Na = 115
        V_L = 10
        dVdt = (1/Cm) * ((g_k * (n**4) * (V_k - V)) + (g_Na * (m**3) * h * (V_Na - V)) + (g_L * (V_L - V)) + self.Iext)
        dndt = alpha_n * (1 - n) - beta_n * n
        dmdt = alpha_m * (1 - m) - beta_m * m
        dhdt = alpha_h * (1 - h) - beta_h * h
        return np.array([dVdt, dndt, dmdt, dhdt])
    
    def G(self, U, t):
        C_m = 1.0
        return np.diag([self.sigma/C_m, 0.0, 0.0, 0.0])
        
    def solve(self):
    # Initial conditions
        V0 = 0
        n0 = 0.35
        m0 = 0.06
        h0 = 0.6
        self.t = np.linspace(0.0,self.duration,100000)
        Uzero = np.array([V0, n0, m0, h0])
        self.solution = itoEuler(self.dUdt, self.G, Uzero, self.t)
            
    def plot_time_series(self):
        if self.solution is not None:
            fig, (ax1, ax2) = plt.subplots(2)
            fig.suptitle("Time series for Iext = "+str(self.Iext)+" and sigma = " +str(self.sigma))
            ax1.plot(self.t, self.solution[:, 0], label = 'V')
            ax1.legend(loc="upper right")
            ax1.set_title('V vs t')
            ax1.set_ylabel("Membrane potential (in mV)")
            ax2.set_title('n, m, h vs t')
            ax2.plot(self.t, self.solution[:, 1], 'tab:red', label = 'n')
            ax2.plot(self.t, self.solution[:, 2], 'tab:orange', label = 'm') 
            ax2.plot(self.t, self.solution[:, 3], 'tab:green', label = 'h')
            ax2.legend(loc="upper right")
            ax2.set_xlabel("Time (in msec)")
            ax2.set_ylabel("Activation")
            fig.subplots_adjust(hspace=.5)
            plt.savefig('graphs/'+'5A_'+self.question_no+'_time_series.png')
        else:
            raise RuntimeError("Equations have not been solved: call solve() method")
            
    def plot_phase_portrait(self): 
        if self.solution is not None:
            plt.clf()
            plt.xlim(self.solution[:,0].min() - 8.0, self.solution[:,0].max() + 8.0)
            plt.ylim(self.solution[:,1].min() - .8, self.solution[:,1].max() + .8)
            plt.plot(self.solution[:,0], self.solution[:,1], color = 'r')
            plt.title('Phase Potrait', fontsize = 14)
            plt.xlabel('V (in mV)')
            plt.ylabel('n (in mV)')
            plt.title("Phase Portrait for Iext = "+str(self.Iext)+" and sigma = " + str(self.sigma))
            plt.savefig('graphs/'+'5A_'+self.question_no+'_phase_portrait.png')
        else:
            raise RuntimeError("Equations have not been solved: call solve() method")
     
    def firing_rate(self):
        if self.solution is not None:
            peaks, _ = find_peaks(self.solution[:,0],height=50.0,prominence=1.5)
            return len(peaks)/self.t[-1]
        else:
            raise RuntimeError("Equations have not been solved: call solve() method")
    
if __name__ == "__main__":
    path = "graphs"
    if not os.path.exists(path):
        os.makedirs(path)
    else:    
        files = glob.glob('/graphs/*')
        for f in files:
            os.remove(f)
    
    HH_plotter = stochastic_HH(5.2, 0.0, question_no="a.i", duration=200.0)
    HH_plotter.solve()
    HH_plotter.plot_time_series()
    HH_plotter.plot_phase_portrait()
    HH_plotter = stochastic_HH(5.2, 0.2, question_no="a.ii", duration=200.0)
    HH_plotter.solve()
    HH_plotter.plot_time_series()
    HH_plotter.plot_phase_portrait()
    HH_plotter = stochastic_HH(5.2, 2.5, question_no="a.iii", duration=200.0)
    HH_plotter.solve()
    HH_plotter.plot_time_series()
    HH_plotter.plot_phase_portrait()
    HH_plotter = stochastic_HH(5.2, 5.0, question_no="a.iv", duration=200.0)
    HH_plotter.solve()
    HH_plotter.plot_time_series()
    HH_plotter.plot_phase_portrait()
    HH_plotter = stochastic_HH(6.6, 0.0, question_no="b.i", duration=200.0)
    HH_plotter.solve()
    HH_plotter.plot_time_series()
    HH_plotter.plot_phase_portrait()
    HH_plotter = stochastic_HH(6.6, 0.2, question_no="b.ii", duration=200.0)
    HH_plotter.solve()
    HH_plotter.plot_time_series()
    HH_plotter.plot_phase_portrait()
    HH_plotter = stochastic_HH(6.6, 2.5, question_no="b.iii", duration=200.0)
    HH_plotter.solve()
    HH_plotter.plot_time_series()
    HH_plotter.plot_phase_portrait()
    HH_plotter = stochastic_HH(6.6, 5.0, question_no="b.iv", duration=200.0)
    HH_plotter.solve()
    HH_plotter.plot_time_series()
    HH_plotter.plot_phase_portrait()
    
    sigma_range = np.arange(0.0, 4.1, 0.1)
    firing_rates = []
    for sigma in tqdm(sigma_range):
        firing_rates_sigma = []
        for i in range(25):
            HH_plotter = stochastic_HH(6.6, sigma, duration = 300.0)
            HH_plotter.solve()
            firing_rates_sigma.append(HH_plotter.firing_rate())
        firing_rates.append(np.mean(firing_rates_sigma))
    firing_rates = np.array(firing_rates)
    plt.clf()
    plt.plot(sigma_range, firing_rates, color = 'blue')
    plt.xlabel(unicodeit.replace('$\sigma$')+' (noise amplitude)')
    plt.ylabel('N (mean firing rate) (in 1/ms)')
    plt.title("Mean Firing Rate vs Noise amplitude ("+unicodeit.replace('I_\{ext\}') + " = 6.6)")
    plt.savefig('graphs/'+'5A_d.png')
