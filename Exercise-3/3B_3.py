import numpy as np
import matplotlib.pyplot as plt
import unicodeit

if __name__ == "__main__":
    a = 0.5
    b = 0.5
    epsl = 0.01
    c = np.linspace(0,100,500)
    lambda_1 = 0.5*(epsl*c-a+np.sqrt((epsl*c-a)**2 + 4*epsl*a*c - 4*epsl*b))
    lambda_2 = 0.5*(epsl*c-a-np.sqrt((epsl*c-a)**2 + 4*epsl*a*c - 4*epsl*b))
    plt.plot(c, lambda_1)
    plt.plot(c, lambda_2)
    plt.ylabel(unicodeit.replace('\lambda'))
    plt.xlabel('c')
    plt.show()