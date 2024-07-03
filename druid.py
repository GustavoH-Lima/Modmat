import numpy as np
import math as m
from numpy.linalg import eig, inv
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Druid stone parameters
# halv-ellipsoid shape
b1 = 7.5 #cm
b2 = 2
b3 = 1.5
# mass of ellipsoid and p masses
me = 18 #grams
mp = 3

g = 981 #gravity cm/s^2

r0 = [6, 6/5] #p masses positions
x0 = r0[0]
y0 = r0[1]

# Initial conditions
gamma = np.array([0, 0, 1])
omega = np.array([0, 0, 0.2]) #initial angular velocity (rad/s)
time_step = 0.01  # time step for the simulation

# Simulation duration
simulation_time = 10.0

def calculate_inertia_tensor():
    i_11 = (me/5)*(b2**2+b3**2)+2*mp*y0**2-(9/64)*(me**2*b3**2/(me+2*mp))
    i_22 = (me/5)*(b3**2+b1**2)+2*mp*x0**2-(9/64)*(me**2*b3**2/(me+2*mp))
    i_33 = (me/5)*(b1**2+b2**2)+2*mp*(x0**2+y0**2)
    i_12 = 2*mp*x0*y0
    Im = np.array([
        [i_11, -i_12, 0],
        [-i_12, i_22, 0],
        [0, 0, i_33]
    ])
    w, v = eig(Im) #eigen values of Im
    delta = m.atan(2*i_12/(i_22-i_11))/2
    i1 = w[0]
    i2 = w[1]
    i3 = w[2]
    I = np.array([
        [i1*m.cos(delta)**2+i2*m.sin(delta)**2, (i1-i2)*m.sin(delta)*m.cos(delta), 0],
        [(i1-i2)*m.sin(delta)*m.cos(delta), i2*m.cos(delta)**2+i1*m.sin(delta)**2, 0],
        [0, 0, i3]
    ])
    return I, inv(I)

def calculate_a_vec():
    g1, g2, g3 = gamma
    a1 = -b1**2*g1/m.sqrt(b1**2*g1**2+b2**2*g2**2+b3**2*g3**2)
    a2 = -b2**2*g2/m.sqrt(b1**2*g1**2+b2**2*g2**2+b3**2*g3**2)
    a3 = -b3**2*g3/m.sqrt(b1**2*g1**2+b2**2*g2**2+b3**2*g3**2)+(3/8)*me*b3/(me+2*mp)
    return np.array([a1, a2, a3])

I, R = calculate_inertia_tensor()
a_vec = calculate_a_vec()
print(a_vec)