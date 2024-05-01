import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math

"""
def gating_variables(V):
    e = math.e
    return [
        (25 - V)/(10*(e**((25-V)/10) - 1)),
        4 * (e ** (-V/18)),
        (7/100)*(e ** (-V/20)),
        1/(1 + e ** ((30 - V)/10)),
        (10 - V)/(100*(e**((10-V)/10) - 1)),
        (1/8)* (e ** (-V/80))    
    ]
"""

Iext = 6.8
e = math.e

def dUdt(U, t):
    """
    Defines the differential equations for the HH model for neuron.

    Arguments:
        w :  vector of the state variables:
                w = [V,n,m,h]
        t :  time
    """
    
    V, n, m, h = U
    alpha_m = (25 - V)/(10*(e**((25-V)/10) - 1))
    beta_m = 4 * (e ** (-V/18))
    alpha_h = (7/100)*(e ** (-V/20))
    beta_h = 1/(1 + e ** ((30 - V)/10))
    alpha_n = (10 - V)/(100*(e**((10-V)/10) - 1))
    beta_n = (1/8)* (e ** (-V/80)) 
    Cm = 1
    g_dash_k = 36
    g_dash_Na = 120
    g_L = 0.3
    V_k = -12
    V_Na = 115
    V_L = 10
    dVdt = (1/Cm) * ((g_dash_k * (n**4) * (V_k - V)) + (g_dash_Na * (m**3) * h * (V_Na - V)) + (g_L * (V_L - V)) + Iext)
    dndt = alpha_n * (1 - n) - beta_n * n
    dmdt = alpha_m * (1 - m) - beta_m * m
    dhdt = alpha_h * (1 - h) - beta_h * h
    # Create f = (V',n',m',h'):
    return [dVdt, dndt, dmdt, dhdt]
    


if __name__ == "__main__":
    # Initial conditions
    V0 = 70
    n0 = 0.15
    m0 = 0.02
    h0 = 0.4
    
    #time
    t = np.linspace(0,200,100)
    
    #pack the initial conditions
    Uzero = [V0, n0, m0, h0]
    
    #call the ODE solver
    solution = odeint(dUdt, Uzero, t)
    
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Problem 4')
    ax1.plot(t, solution[:, 0], label = 'V')
    ax1.legend(loc="upper right")
    ax1.set_title('V vs t')
    ax1.set_ylabel("Membrane potential (in mV)")
    ax2.plot(t, solution[:, 1], 'tab:red', label = 'n')
    #ax2.set_title('n vs t')
    ax2.plot(t, solution[:, 2], 'tab:orange', label = 'm') 
    #ax3.set_title('m vs t')
    ax2.plot(t, solution[:, 3], 'tab:green', label = 'h')
    ax2.legend(loc="upper right")
    ax2.set_xlabel("Time (in msec)")
    ax2.set_ylabel("Activation")
    #ax4.set_title('h vs t')

    #for ax in fig.get_axes():
    #    ax.label_outer()
    

    #plt.legend() 
    #plt.xlabel("Time")
    
    plt.show()
    
    
