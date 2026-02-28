
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib.animation import FuncAnimation

'''
1: CREATE GRID
'''
Nx     = 500
x      = np.linspace(0.01, 10.0, Nx)   # range of positions
dx     = x[1] - x[0]

dt     = 1e-4                         # timestep
t_max  = 1.0                          # final time
Nt     = int(t_max / dt)              # number of timesteps
t      = np.linspace(0, t_max, Nt)    # array of time values


'''
2: INITIAL CONDITIONS
'''
nu     = 3                            # viscosity 
x0     = 5                            # initial gaussian profile
std    = 0.5
Sigma0 = np.exp(-(x - x0)**2 / (2 * std**2))

D      = 3 * nu                       # difussion coefficient
u      = - 9 * nu / 2 / x             # velocity
beta   = dt / dx**2 * D               # parameter in implicit method
sqrt_x = np.sqrt(x)                   # factor in the differential equation


print('-------stability criterion--------')
if np.max(np.abs(u)) * dt / dx <= 1:  # check if the CFL criteion is satisfied
    stable = 'yes'
else:
    stable = 'no'


'''
3: IMPLICIT DIFFUSION SOLVER
'''
A     = np.eye(Nx-2) * (1.0 + 2.0 * beta) + np.eye(Nx-2, k=1) * (-beta) + np.eye(Nx-2, k=-1) * (-beta)
def implicit_method(f):
    T         = f.copy()

    return np.linalg.solve(A, T)

'''
4: LAX FRIEDRICH METHOD ADVECTION SOLVER
'''
def lax_friedrichs(f):
    f_i       = np.zeros_like(f, dtype=np.float64)

    f_i[1:-1] = 0.5 * (f[2:] + f[:-2]) - (dt * u[1:-1] / (2 * dx)) * (f[2:] - f[:-2])


    return f_i


'''
5: TIME INTEGRATION
'''
Sigma    = np.zeros(shape=(Nt, Nx), dtype=np.float64)

for i, t_i in enumerate(t):
    print('stable?', stable, 'time =', t_i)
    # begin with the initial condition
    if t_i == 0 and i == 0:
        Sigma[i] = Sigma0.copy()
    else:
        Sigma_i        = np.zeros_like(Sigma0, dtype=np.float64)

        # solve difussion part of the equation
        Sigma_i[1:-1]  = implicit_method(Sigma[i-1][1:-1])
        Sigma_i[0]     = Sigma_i[1]      # outflow boundary conditions
        Sigma_i[-1]    = Sigma_i[-2]

        # solve advection part of the equation
        Sigma_i        = lax_friedrichs(Sigma_i)
        Sigma_i[0]     = Sigma_i[1]      # outflow boundary conditions
        Sigma_i[-1]    = Sigma_i[-2]

        # recover the surface density
        Sigma[i] = Sigma_i

# save simulation output
np.save('output.npy', Sigma)

'''
6: MAKE ANIMATION
'''
# set up the plot
fig, ax = plt.subplots(1, 1, figsize=(4, 3))

# load data
Sigma  = np.load('output.npy')

# choose 100 frames evenly from all timesteps
Nt            = Sigma.shape[0]
num_frames    = 500
frame_indices = np.linspace(0, Nt-1, num_frames, dtype=int)
Sigma_anim    = Sigma[frame_indices]

# plot the initial state
line,  = ax.plot(x, Sigma[0], '-', color='black', lw=1)

ax.set_xlabel('$x$')
ax.set_ylabel(r'$\Sigma\:(x, \tau)$');
ax.set_xlim(0, 10)
ax.set_ylim(0, 1.1)

plt.tight_layout()

def update(frame):
    line.set_ydata(Sigma_anim[frame])
    return line,

anim = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)

anim.save('evolution.mp4', fps=20, dpi=150)