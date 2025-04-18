import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
#import pyknotid
from ctypes.util import find_library
find_library('m')
exit()
# function that returns dy/dt
def model(y, t):
    G = 0.15
    sign = -1
    dydt = sign * np.sqrt((1. - y) ** 2 * y - G ** 2)
    return dydt


# initial condition
y0 = 0.82

# time points
t = np.linspace(4, 7.5, 200)

# solve ODE
y = odeint(model, y0, t)

# plot results
plt.plot(t, y)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.show()
