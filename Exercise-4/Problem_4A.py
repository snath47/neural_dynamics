import numpy as np
import matplotlib.pyplot as plt
import unicodeit
import os

def Iext(t, omega=0.25):
    A = np.less_equal(t, 50) * 0.0 + np.logical_and(np.greater(t,50),np.less_equal(t, 100)) * 100.0 + np.logical_and(np.greater(t,100),np.less_equal(t, 150)) * 200.0 + np.logical_and(np.greater(t,150),np.less_equal(t, 200)) * 0.0
    return A * np.cos(omega * t)
    

def simulate_LIF(t, dt, omega, V_rest = -70.0):
    V_threshold = -55.0
    V_reset = -75.0
    R_m = 1
    tau_m = 20.0
    V = [V_rest,]
    I = []
    for it in range(len(t)-1):
        if V[it] >= V_threshold:
            V[it] = V_reset
        dV = (-(V[it]-V_rest) + R_m * Iext(t[it], omega))*(dt/tau_m)
        V.append(V[it] + dV)
        I.append(Iext(t[it]))
    I.append(Iext(t[-1]))        
    return np.array(V), np.array(I)
        
    

if __name__ == "__main__":
    path = "graphs"
    if not os.path.exists(path):
        os.makedirs(path)
    omega_list = [.25, 1.0, 10.0]
    dt = 0.1
    t = np.arange(0, 200, dt)
    for i in range(len(omega_list)):
        V, I = simulate_LIF(t, dt, omega_list[i])
        fig, (ax1, ax2) = plt.subplots(2)
        plt.suptitle('Time Series ('+ unicodeit.replace("\omega")+' = '+str(omega_list[i])+')', fontsize = 14)
        ax1.plot(t, V)
        ax1.set_title('V vs t')
        ax1.set_ylabel("Membrane potential (in mV)")
        ax2.set_title(unicodeit.replace("$I_\{ext\}$")+ ' vs t')
        ax2.plot(t, I, 'tab:red')
        ax2.set_xlabel("Time (in msec)")
        ax2.set_ylabel("External Current (in " + unicodeit.replace("\mu A/cm^2") +")")
        fig.subplots_adjust(hspace=.5)
        plt.savefig('graphs/'+'LIF_time_series_4A_'+str(i+1))