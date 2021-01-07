import numpy as np
import matplotlib.pyplot as plt
#import scipy.constants as c
import time

def distance(par1, par2):
    return np.sqrt((par1[1] - par2[1])**2 + (par1[2] - par2[2])**2 + (par1[3] - par2[3])**2)

def acceleration(par1, par2, rad):
    global G
    acc = lambda p1, p2: G*par1[0]*(p1 - p2)/rad**3
    return [acc(par1[1], par2[1]), acc(par1[2], par2[2]), acc(par1[3], par2[3])]

def potential_energy(par1, par2, rad):
    global G
    return -G*par1[0]*par2[0]/(2*rad) # Where did this factor of 2 come from?

def velocity(par, dt):
    vel = lambda v, a: v + a*dt
    return [vel(par[4], par[7]), vel(par[5], par[8]), vel(par[6], par[9])]

def kinetic_energy(par):
    return 0.5*par[0]*(par[4]**2 + par[5]**2 + par[6]**2)
    
def position(par, dt):
    pos = lambda p, v: p + v*dt
    return [pos(par[1], par[4]), pos(par[2], par[5]), pos(par[3], par[6])]


# format: m, x, y, z, vx, vy, vz
# data = [[5, -0.2, 0.2, 0, 5, 5, 0],
#         [1, 0.3, 0, 0, 0, 0, 0],
#         [1, 0, 0.3, 0, 0, 0, 0],
#         [1, 0, -0.3, 0, 0, 0, 0],
#         [10, 0, 0, 0, 2, 4, 0]]

# data = [np.random.random(7) for n in range(4)] # random position, velocity and mass

data = [[1, 0, 0, 0, 0, 0, 0],
        [1, 0.5, 0, 0, 0, 1.63, 0],
        [1, 0.25, 0.25, 0, 0.8, 0.8, 0]]

G = 1
dt_base = 6E-5
variable_step = True
steps = int(1E5)
particles = len(data)
dimensions = len(data[0]) + 6 # 3 for accelerations, then potential, kinetic and total energy

start = time.perf_counter()
# Set up a big 3-d array to store everything
states = np.zeros((particles, steps + 1, dimensions))
# Insert initial values from data list and add object mass to each row
for i in range(particles):
    states[i][:,0] = (np.zeros((1, steps + 1)) + data[i][0]) # Make list of object mass then add this to the mass column in states array
    states[i][0][1:7] = data[i][1:7]
end = time.perf_counter()
print(end - start)

dt = dt_base
start = time.perf_counter()
for t in range(steps):
    rad = []
    for i in range(particles):
        for j in range(particles):
            if i != j:
                rad.append(distance(states[j, t], states[i, t]))
                acc = acceleration(states[j, t], states[i, t], rad[-1])
                states[i, t + 1, 7] += acc[0]
                states[i, t + 1, 8] += acc[1]
                states[i, t + 1, 9] += acc[2]
                states[i, t + 1, 10] += potential_energy(states[j, t], states[i, t], rad[-1])
        

        states[i, t + 1, 4:7] = velocity(states[i, t], dt)
        
        states[i, t + 1, 11] = kinetic_energy(states[i, t])

        states[i, t + 1, 1:4] = position(states[i, t], dt)
        
        states[i, t + 1, 12] = states[i, t, 11] + states[i, t, 10]
        
    if variable_step is True:
        dt = min(rad)*dt_base #Try out some different functions here! Also, this doesn't work when the radius is always bigger than 1 - might need to fix that. 
end = time.perf_counter()
print(end - start)


fig = plt.figure()
ax = plt.axes()
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
for i in range(particles):
    ax.plot(states[i, :, 1], states[i, :, 2])
    ax.scatter(states[i, 0, 1], states[i, 0, 2])

E_total = []
for t in range(steps + 1):
    E = 0
    for i in range(particles):
        E += states[i][t][12]
    E_total.append(E)
    
E_k = []
for t in range(steps + 1):
    E = 0
    for i in range(particles):
        E += states[i][t][11]
    E_k.append(E)

E_p = []
for t in range(steps + 1):
    E = 0
    for i in range(particles):
        E += states[i][t][10]
    E_p.append(E)

fig = plt.figure()
ax = plt.axes()
ax.plot(np.arange(steps + 1), E_total)
ax.plot(np.arange(steps + 1), E_k)
ax.plot(np.arange(steps + 1), E_p)

# fig = plt.figure()
# ax = plt.axes(projection="3d")
# ax.plot3D(state[0][:, 0], state[0][:, 1], state[0][:, 2])
# ax.plot3D(state[1][:, 0], state[1][:, 1], state[1][:, 2])

# start = time.perf_counter()
# from celluloid import Camera
# fig = plt.figure()
# ax = plt.axes()
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# camera = Camera(fig)
# for i in range(steps):
#     plt.scatter(states[0][i][0], states[0][i][1], color = "m")
#     plt.scatter(states[1][i][0], states[1][i][1], color = "g")
#     plt.scatter(states[2][i][0], states[2][i][1], color = "r")
#     plt.scatter(states[3][i][0], states[3][i][1], color = "b")
#     plt.scatter(states[4][i][0], states[4][i][1], color = "k")
#     camera.snap()
# animation = camera.animate(interval = 1/60*(10**-3))
# animation.save("celluloid_minimal.gif", writer = "pillow")
# end = time.perf_counter()
# print(end - start)