import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math

def gating_variables(V):
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

def dUdt(U, t, Iext):
    """
    Defines the differential equations for the HH model for neuron.

    Arguments:
        U :  vector of the state variables:
                U = [V,n,m,h]
        t :  time
        Iext: external current as parameter
        
    Returns:
        an array containing the differentials of the state variables
    """
    
    V, n, m, h = U
    alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = gating_variables(V)
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

    return [dVdt, dndt, dmdt, dhdt]
    


if __name__ == "__main__":
    # Initial conditions
    V0 = 70
    n0 = 0.25
    m0 = 0.07
    h0 = 0.65
    
    #time
    t = np.linspace(0,200,100)
    
    #pack the initial conditions
    Uzero = [V0, n0, m0, h0]
    
    #set Iext
    Iext = 5.2
    
    #call the ODE solver
    solution = odeint(dUdt, Uzero, t, args=(Iext,))
    
    #generate the plots
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Problem 2')
    ax1.plot(t, solution[:, 0], label = 'V')
    ax1.legend(loc="upper right")
    ax1.set_title('V vs t')
    ax1.set_ylabel("Membrane potential (in mV)")
    ax2.set_title('n, m, h vs t')
    ax2.plot(t, solution[:, 1], 'tab:red', label = 'n')
    ax2.plot(t, solution[:, 2], 'tab:orange', label = 'm') 
    ax2.plot(t, solution[:, 3], 'tab:green', label = 'h')
    ax2.legend(loc="upper right")
    ax2.set_xlabel("Time (in msec)")
    ax2.set_ylabel("Activation")
    plt.show()
    
    
