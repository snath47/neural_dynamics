import numpy as np
import matplotlib.pyplot as plt
import unicodeit

if __name__ == "__main__":
    a = np.linspace(0, 1, 100)
    xs_1 = ((a + 1) + np.sqrt(a**2.0 - a + 1.0))/3.0
    ys_1 = xs_1 * (xs_1-a) * (1-xs_1)
    xs_2 = ((a + 1) - np.sqrt(a**2.0 - a + 1.0))/3.0
    ys_2 = xs_2 * (xs_2-a) * (1-xs_2)
    plt.plot(a, 3*xs_1 - a - 1)
    plt.plot(a, 3*xs_2 - a - 1)
    plt.ylabel(unicodeit.replace('3x_s-a-1'))
    plt.xlabel('a')
    plt.show()