import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Parameters
N = 30
T = 1000
dt = 0.1
steps = int(T / dt)
epsilon = 0.05
beta = 0.756
I = 0.001
gap_junction_strength = 0.25
noise_amplitude = 0.005
time_delay_0 = 0
time_delay_5 = 5
alpha = 0.5 #parameter to set from (0, 1)


G = nx.generators.random_graphs.gnp_random_graph(N, 0.8)
adjacency_matrix = nx.adjacency_matrix(G).toarray()
np.fill_diagonal(adjacency_matrix, 0)

# Initialize variables
v = np.random.rand(N)
w = np.random.rand(N)
v_history_0 = np.zeros((steps, N))
v_history_5 = np.zeros((steps, N))

# Simulation function
def simulate(time_delay):
    v = np.random.rand(N)
    w = np.random.rand(N)
    v_history = np.zeros((steps, N))
    
    for t in range(steps):
        eta = noise_amplitude * np.random.randn(N)
        
        if t >= time_delay:
            v_delayed = v_history[t-time_delay] if time_delay > 0 else v
        else:
            v_delayed = v
        
        dv = (v - (v**3)/3 - w + I + gap_junction_strength * np.dot(adjacency_matrix, v_delayed - v))
        dw = epsilon*(v + alpha - beta * w)
        
        v = v + dv * dt + eta * np.sqrt(dt)
        w = w + dw * dt
        
        v_history[t] = v
    
    return v_history

# Run simulations
v_history_0 = simulate(time_delay_0)
v_history_5 = simulate(time_delay_5)

# Plot results
neuron_indices = [0, 4, 9, 14, 24]
time = np.linspace(0, T, steps)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols = 2, figsize=(12, 6))

for idx in neuron_indices:
    ax1.plot(time, v_history_0[:, idx], label=f'Neuron {idx+1} (delay=0)')

ax1.set_title('Membrane Potential of Selected Neurons (Time Delay = 0)')
ax1.set_xlabel('Time')
ax1.set_ylabel('Membrane Potential (v)')
ax1.legend(loc="upper right")

for idx in neuron_indices:
    ax2.plot(time, v_history_5[:, idx], label=f'Neuron {idx+1} (delay=5)')

ax2.set_title('Membrane Potential of Selected Neurons (Time Delay = 5)')
ax2.set_xlabel('Time')
ax2.set_ylabel('Membrane Potential (v)')
ax2.legend(loc="upper right")
plt.show()
