import numpy as np
import matplotlib.pyplot as plt
import unicodeit

if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(1,2)
    a = np.linspace(0, 1, 100)
    xs_1 = ((a + 1) + np.sqrt(a**2.0 - a + 1.0))/3.0
    ys_1 = xs_1 * (xs_1-a) * (1-xs_1)
    xs_2 = ((a + 1) - np.sqrt(a**2.0 - a + 1.0))/3.0
    ys_2 = xs_2 * (xs_2-a) * (1-xs_2)
    ax1.plot(a, xs_1)
    ax1.plot(a, xs_2)
    ax2.plot(a, ys_1)
    ax2.plot(a, ys_2)
    ax1.set_ylabel(unicodeit.replace('x_s'))
    ax2.set_ylabel(unicodeit.replace('y_s'))
    ax1.set_xlabel('a')
    ax2.set_xlabel('a')
    plt.show()