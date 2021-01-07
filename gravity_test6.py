import numpy as np
import matplotlib.pyplot as plt
import random as rn
#import scipy.constants as c
import time


 # Define all the functions to be used later on.
def distance(par1, par2):
    """
    
    Parameters
    ----------
    par1 : numpy.ndarray
        Current state of particle 1 in format:
        [m, x, y, z, vx, vy, vz, ax, ay, az, Ep, Ek, Et].
    par2 : numpy.ndarray
        Current state of particle 2 in format:
        [m, x, y, z, vx, vy, vz, ax, ay, az, Ep, Ek, Et].

    Returns
    -------
    numpy.float64
        Straight line distance between particle 1 and particle 2. 

    """
    return np.sqrt((par1[1] - par2[1])**2 + (par1[2] - par2[2])**2 + (par1[3] - par2[3])**2)


def acceleration(par1, par2, rad):
    """
    
    Parameters
    ----------
    par1 : numpy.ndarray
        Current state of particle 1 in format:
        [m, x, y, z, vx, vy, vz, ax, ay, az, Ep, Ek, Et].
    par2 : numpy.ndarray
        Current state of particle 2 in format:
        [m, x, y, z, vx, vy, vz, ax, ay, az, Ep, Ek, Et].
    rad : numpy.float64
        Straight line distance between particle 1 and particle 2. 

    Returns
    -------
    list
        Components of particle 1's acceleration due to particle 2 in format: [ax, ay, az].

    """ 
    acc = lambda p1, p2: G*par1[0]*(p1 - p2)/rad**3
    return [acc(par1[1], par2[1]), acc(par1[2], par2[2]), acc(par1[3], par2[3])]


def potential_energy(par1, par2, rad):
    """
    
    Parameters
    ----------
    par1 : numpy.ndarray
        Current state of particle 1 in format:
        [m, x, y, z, vx, vy, vz, ax, ay, az, Ep, Ek, Et].
    par2 : numpy.ndarray
        Current state of particle 2 in format:
        [m, x, y, z, vx, vy, vz, ax, ay, az, Ep, Ek, Et].
    rad : numpy.float64
        Straight line distance between particle 1 and particle 2. 

    Returns
    -------
    numpy.float64
        Potential energy of particle 1 due to particle 2.

    """
    return -G*par1[0]*par2[0]/(2*rad) # Where did this factor of 2 come from? It's because I calculate the potential energy between particles i and j, but also between j and i so total potential is doubled! Fix this, it slows things down a lot


def velocity(par, dt):
    """
    
    Parameters
    ----------
    par : numpy.ndarray
        Current state of the particle in format:
        [m, x, y, z, vx, vy, vz, ax, ay, az, Ep, Ek, Et].
    dt : float
        Time-step length.

    Returns
    -------
    list
        Components of the particle's velocity in format: [vx, vy, vz].

    """
    vel = lambda v, a: v + a*dt
    return [vel(par[4], par[7]), vel(par[5], par[8]), vel(par[6], par[9])]


def kinetic_energy(par):
    """
    
    Parameters
    ----------
    par : numpy.ndarray
        Current state of the particle in format:
        [m, x, y, z, vx, vy, vz, ax, ay, az, Ep, Ek, Et].

    Returns
    -------
    numpy.float64
        Kinetic energy of the particle.

    """
    return 0.5*par[0]*(par[4]**2 + par[5]**2 + par[6]**2)

    
def position(par, dt):
    """
    
    Parameters
    ----------
    par : numpy.ndarray
        Current state of the particle in format:
        [m, x, y, z, vx, vy, vz, ax, ay, az, Ep, Ek, Et].
    dt : float
        Time-step length.

    Returns
    -------
    list
        Components of the particle's position in format: [x, y, z].

    """
    pos = lambda p, v: p + v*dt
    return [pos(par[1], par[4]), pos(par[2], par[5]), pos(par[3], par[6])]



logistic = lambda x, x0, k : 1/(1 + np.exp(-k*(x - x0)))


# Data format: [[m1, x1, y1, z1, vx1, vy1, vz1], [m2, x2, y2, z2, vx2, vy2, vz2]...].

# data = [[5, -0.2, 0.2, 0, 5, 5, 0],
#         [1, 0.3, 0, 0, 0, 0, 0],
#         [1, 0, 0.3, 0, 0, 0, 0],
#         [1, 0, -0.3, 0, 0, 0, 0],
#         [10, 0, 0, 0, 2, 4, 0]]

# data = [[rn.random() for n in range(7)] for m in range(6)] # random position, velocity and mass

data = [[1, 0, 0, 0, 0, 0, 0],
        [1, 0.5, 0, 0, 0, 1.63, 0],
        [1, 0.25, 0.25, 0, 0.8, 0.8, 0]]

# data = [[1, 0.2, 0.05, 0, -0.5, 0, 0], [1, -0.2, -0.05, 0, 0.5, 0, 0]]

 # Add zeros for the extra dimensions to each particle in data.
extra_dims = 6 # x, y and z accelerations; potential, kinetic and total energy
for particle in data:
    for n in range(extra_dims):
        particle.append(0)

 # Create an array of initial particle states from data (in a list to make it 3-d).
state = np.array([data])
state_zeros = np.zeros(np.shape(state))

 # Define numbers of particles and dimensions.
particles = np.shape(state)[1]
dimensions = np.shape(state)[2]


 # Define timing stuff and whatever else is needed.
G = 1
dt_base = dt = 2E-4 # 1E-5 and lower can get very slow!
t_duration = 0.3
variable_step = True


# Initialise timing stuff and start the loop.
start = time.perf_counter() # Start a timer.
step_times = [dt]
t_elapsed = 0
t = 0
while t_elapsed <= t_duration:
    
    state = np.concatenate([state, state_zeros]) # Add state_zeros as a 'layer' to state.
    rad = [] # Define a list to contain all the radii at this step.
    
     # Loop over all pairs of particles i and j.
    for i in range(particles):
        
        state[t + 1, i, 0] = state[0, i, 0] # Add particle masses to the new layer.
        
        for j in range(particles):
            
            if i != j:
                
                 # Calculate radius between i and j.
                rad.append(distance(state[t, j], state[t, i]))
                
                 # Calculate acceleration of i from j as a list: [ax, ay, az].
                acc = acceleration(state[t, j], state[t, i], rad[-1])
                
                 # Add accelerations to particle i's current entry in state.
                state[t, i, 7] += acc[0]
                state[t, i, 8] += acc[1]
                state[t, i, 9] += acc[2]
                
                 # Calculate potential energy of i from j and add this to i's current
                 # entry in state.
                state[t, i, 10] += potential_energy(state[t, j], state[t, i], rad[-1])
        
         # Calculate i's new velocity and add to its next entry in state.
        state[t + 1, i, 4:7] = velocity(state[t, i], dt)
        
         # Calculate i's kinetic energy and add to its current entry in state.
        state[t, i, 11] = kinetic_energy(state[t, i])

         # Calculate i's position and add to its next entry in state
        state[t + 1, i, 1:4] = position(state[t, i], dt)
        
         # Calculate i's total energy (potential + kinetic) and add to it's current entry
         # in state.
        state[t, i, 12] = state[t, i, 11] + state[t, i, 10]
    
     # Add current dt to the elapsed time, print it, then add this to the step_times list.
    t_elapsed += dt
    print(t_elapsed)
    step_times.append(t_elapsed)
    t += 1 # Using a while loop need to iterate t manually.
    
     # vary the length of dt based on the current shortest distance between any two
     # particles, min(rad).
    if variable_step is True:
        if max(acc) >= 150 or min(acc) <= -150:
            dt = (min(rad)*dt_base)**1.16
        else:
            dt = dt_base
        
end = time.perf_counter() # End the timer.
print(end - start)


 # Calculate potential, kinetic and total energy for the final step.
for i in range(particles):
    for j in range (particles):
        if i != j:
            rad = distance(state[-1, i], state[-1, j])
            state[-1, i, 10] += potential_energy(state[-1, i], state[-1, j], rad)
    state[-1, i, 11] = kinetic_energy(state[-1, i])
    state[-1, i, 12] = state[-1, i, 10] + state[-1, i, 11]


 # Plot things
fig = plt.figure()
ax = plt.axes()
# ax.set_xlim(-0.1, 0.6)
# ax.set_ylim(-0.1, 0.6)
for i in range(particles):
    ax.plot(state[:, i, 1], state[:, i, 2])
    ax.scatter(state[0, i, 1], state[0, i, 2])

fig = plt.figure()
ax = plt.axes()
E_total = []
for i in range(particles):
    ax.plot(step_times, state[:, i, 12])
for t in range(len(step_times)):
    E = 0
    for i in range(particles):
        E += state[t, i, 12]
    E_total.append(E)
ax.plot(step_times, E_total, color = "k")

print(max(E_total) - min(E_total))


# fig = plt.figure()
# ax = plt.axes(projection="3d")
# ax.plot3D(state[0][:, 0], state[0][:, 1], state[0][:, 2])
# ax.plot3D(state[1][:, 0], state[1][:, 1], state[1][:, 2])