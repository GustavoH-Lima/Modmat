import numpy as np
import math as m
from numpy.linalg import eig, inv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

# Druid stone parameters
# halv-ellipsoid shape
b1 = 7.5 #cm
b2 = 2
b3 = 1.5
b = np.array([b1, b2, b3])
# mass of ellipsoid and p masses
me = 18 #grams
mp = 3
mass = me + 2*mp

g = 981 #gravity cm/s^2

r0 = [6, 6/5] #p masses positions: r0 and -r0
x0 = r0[0]
y0 = r0[1]
animation_xyz0 = np.array([r0[0], r0[1], 0])

### Simulation Variables
# Initial conditions
# FIXED FRAME VECTORS
gamma = np.array([0, 1/1000, 3*m.sqrt(111111)/1000])
alpha = np.array([1, 0, 0]) #auxiliar vectors to build animation's rotation matrix
beta = np.array([0, 1, 0])
#angular velocity omega
omega = np.array([0, 0, -100]) #initial angular velocity (rad/s)
# M angular momentum in fixed frame
M = np.array([0,0,0])

time_step = 0.01  # time step for the simulation

# Simulation duration
simulation_time = 20 #amount of frames

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

def calculate_a_vec(gamma):
    g1, g2, g3 = gamma
    div = m.sqrt(b1**2*g1**2+b2**2*g2**2+b3**2*g3**2)
    a1 = -b1**2*g1/div
    a2 = -b2**2*g2/div
    a3 = -b3**2*g3/div+(3/8)*me*b3/(me+2*mp)
    return np.array([a1, a2, a3])

a = calculate_a_vec(gamma)

def calculate_rotation_matrix(alpha, beta, gamma):
    Rot = np.column_stack((alpha, beta, gamma))
    return Rot

def update_fixed_frame_vectors():
    #apply d gamma/dt = omega X gamma
    #and renormalize
    global gamma
    global alpha
    global beta
    dgamma = time_step*(np.cross(omega, gamma))
    dalpha = time_step*(np.cross(omega, alpha))
    dbeta = time_step*(np.cross(omega, beta))
    gamma = gamma+dgamma
    alpha = alpha+dalpha
    beta = beta+dbeta
    gamma = gamma*(1/np.linalg.norm(gamma))
    alpha = alpha*(1/np.linalg.norm(alpha))
    beta = beta*(1/np.linalg.norm(beta))
    #reorthogonalize
    beta = np.cross(gamma, alpha)
    alpha = np.cross(beta, gamma)

I, R = calculate_inertia_tensor()


def calculate_M(a, omega):
    return I @ omega + np.cross(mass*a, np.cross(omega,a))

# Define the differential equations
def equations(t, y, omega, mass):
    M = y[:3]
    gamma = y[3:6]
    alpha = y[6:9]
    beta = y[9:12]
    a = y[12:15]
    
    # Reshape vectors
    M = M.reshape((3,))
    gamma = gamma.reshape((3,))
    alpha = alpha.reshape((3,))
    beta = beta.reshape((3,))
    a = a.reshape((3,))

    #calculate omega value
    # w = J^-1 M
    # J = I + ma^2I-maa^T
    J = I + (mass*(np.linalg.norm(a))**2)*I - mass*(a @ np.transpose(a))
    omega = np.linalg.inv(J) @ M
    # Compute the derivatives
    # should be 
    #     dMdt = np.cross(M, omega) + mass * np.cross(dadt, np.cross(omega, a)) + mass * g * np.cross(a, gamma)
    #                                                   ^~~~how to get da/dt ?
    dMdt = np.cross(M, omega) + mass * np.cross(a, np.cross(omega, a)) + mass * g * np.cross(a, gamma)
    dgamma_dt = np.cross(omega, gamma)
    dalpha_dt = np.cross(omega, alpha)
    dbeta_dt = np.cross(omega, beta)
    #dadt = b*b*(dgammadt/norm(b*gamma) + gamma*(-1/(norm(b*gamma)**2) * gamma dot b*b / norm(b*gamma) )
    dadt = b*b*(dgamma_dt/np.linalg.norm(b*gamma)) + gamma*( (-1/(np.linalg.norm(b*gamma))**2)*(np.dot(gamma, b*b))/np.linalg.norm(b*gamma) )
    
    return np.concatenate([dMdt, dgamma_dt, dalpha_dt, dbeta_dt, dadt])

gamma0 = gamma
alpha0 = alpha
beta0 = beta
a0 = calculate_a_vec(gamma)
M0 = calculate_M(a0, omega)

#initial state
y0 = np.concatenate([M0, gamma0, alpha0, beta0, a0])

# Time span for the solution
total_time = 20
fps = 20
total_frames = total_time*fps
detail = 1000
t_span = (0, total_time)
t_eval = np.linspace(0, total_time, detail)
#solve
solution = solve_ivp(equations, t_span, y0, args=(omega, mass), t_eval=t_eval, method='RK45')


# Extract the solutions
M_solution = solution.y[:3].T
gamma_solution = solution.y[3:6].T
alpha_solution = solution.y[6:9].T
beta_solution = solution.y[9:12].T

# Plot the solutions
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(solution.t, M_solution)
plt.title('M(t) Components')
plt.xlabel('Time')
plt.ylabel('M')
plt.legend(['M_x', 'M_y', 'M_z'])

plt.subplot(2, 1, 2)
plt.plot(solution.t, gamma_solution)
plt.title('γ(t) Components')
plt.xlabel('Time')
plt.ylabel('γ')
plt.legend(['γ_x', 'γ_y', 'γ_z'])

plt.tight_layout()
plt.show()


## ANIMAÇÃO -------------------

# Criação dos pontos para o elipsoide
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 50)

x = b1 * np.outer(np.cos(u), np.sin(v))
y = b2 * np.outer(np.sin(u), np.sin(v))
z = b3 * np.outer(np.ones_like(u), np.cos(v))

# Filtrar para apenas a metade inferior do elipsoide
z_mask = z > 0
x[z_mask] = np.nan
y[z_mask] = np.nan
z[z_mask] = np.nan

cm = [0,0,-(3/8)*(me*b3)/(2*mp*me)]
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

def rotate_ellipsoid(Rot):
    # Aplicar rotação
    xyz = np.vstack([x.flatten(), y.flatten(), z.flatten()])
    xyz_rotated = Rot @ xyz
    x_rot = xyz_rotated[0, :].reshape(x.shape)
    y_rot = xyz_rotated[1, :].reshape(y.shape)
    z_rot = xyz_rotated[2, :].reshape(z.shape)

    # Aplicar rotação na posição das massas auxiliares p
    global animation_xyz0
    animation_xyz0 = Rot @ np.array([r0[0],r0[1], 0])
    
    return x_rot, y_rot, z_rot



def update(frame):
    ax.clear()
    # Remover os indicadores dos números nos eixos
    ax.set_box_aspect([b1, b1, b1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #Remover marcação dos eixos
    # ax.set_yticks([-2,-0.5,0.5,2])
    #ax.set_zticks([-1,0,1])
    ax.axes.set_xlim3d(left=-b1, right=b1) 
    ax.axes.set_ylim3d(bottom=-b1, top=b1) 
    ax.axes.set_zlim3d(bottom=-b1, top=b1) 

    #Update simulation
    #update_fixed_frame_vectors()
    idx = int(total_frames/frame)
    #recalculate rotation matrix
    Rot = calculate_rotation_matrix(alpha_solution[idx], beta_solution[idx], gamma_solution[idx])

    new_ellipse = rotate_ellipsoid(Rot)

    ax.scatter(*cm,color='y')
    ax.scatter(*animation_xyz0, color='k')
    ax.scatter(*(-1*animation_xyz0), color='k')
    #ax.quiver(0,0,0,[b1,0,0],[0,b2,0],[0,0,b3], length=3, normalize=True, arrow_length_ratio=0.21,color='r')
    #ax.quiver(0,0,0,I[0],I[1],I[2], length=2, normalize=True, arrow_length_ratio=0.21,color='g')
    ax.quiver(0,0,0,*gamma_solution[idx], length=2, arrow_length_ratio=0.21, color='pink')
    ax.quiver(0,0,0,*alpha_solution[idx], length=2, arrow_length_ratio=0.21, color='g')
    ax.quiver(0,0,0,*beta_solution[idx], length=2, arrow_length_ratio=0.21, color='r')

    fig.suptitle("Rattleback!")
    surf = ax.plot_surface(*new_ellipse, color='b', alpha=0.6)
    return surf,

#Criar animação
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=1, blit=False)
#writer = animation.PillowWriter(fps=20)
#ani.save("ani.gif", writer=writer)

plt.show()