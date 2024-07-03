import numpy as np
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

# Initial conditions
gamma = [0, 0, 1]   # initial angle (radians)
omega = [0, 0, 0.2]   # initial angular velocity (rad/s)
time_step = 0.01  # time step for the simulation

# Simulation duration
simulation_time = 10.0