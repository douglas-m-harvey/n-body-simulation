import numpy as np
import matplotlib.pyplot as plt
#import scipy.constants as c
import time


# format: x, y, z, vx, vy, vz, m
data = [[-0.2, 0.2, 0, 5, 5, 0, 5],
        [0.3, 0, 0, 0, 0, 0, 1],
        [0, 0.3, 0, 0, 0, 0, 1],
        [0, -0.3, 0, 0, 0, 0, 1],
        [0, 0, 0, 2, 4, 0, 10]]

#data = [np.random.random(7) for n in range(20)] # random position, velocity and mass

mu = 0.1
G = 1
dt = 5E-5
steps = 3000
particles = len(data)
dimensions = len(data[0])

states = np.zeros((particles, steps + 1, dimensions))

for i in range(particles):
    states[i][0] = data[i]

def distance(par1, par2):
    return np.sqrt((par1[0] - par2[0])**2 + (par1[1] - par2[1])**2 + (par1[2] - par2[2])**2)

def acceleration(par1, par2, rad):
    global G
    acc = lambda p1, p2: G*par1[6]*(p1 - p2)/rad**3
    return [acc(par1[0], par2[0]), acc(par1[1], par2[1]), acc(par1[2], par2[2])]

def velocity(par, acc, dt):
    vel = lambda v, a: v + a*dt
    return [vel(par[3], acc[0]), vel(par[4], acc[1]), vel(par[5], acc[2])]
    
def position(par, vel, dt):
    pos = lambda p, v: p + v*dt
    return [pos(par[0], vel[0]), pos(par[1], vel[1]), pos(par[2], vel[2])]


start = time.perf_counter()
for t in range(steps):
    for i in range(particles):
        acc = [0, 0, 0]#find a better way to do this, change the function
        rad = []
        for j in range(particles):
            if i != j:
                rad.append(distance(states[j][t], states[i][t]))
                acc[0] += acceleration(states[j][t], states[i][t], rad[-1])[0]
                acc[1] += acceleration(states[j][t], states[i][t], rad[-1])[1]
                acc[2] += acceleration(states[j][t], states[i][t], rad[-1])[2]

        vel = velocity(states[i][t], acc, dt)
    
        pos = position(states[i][t], vel, dt)
    
        states[i][t + 1] = np.array([pos + vel + [states[i][0][6]]])
end = time.perf_counter()
print(end - start)


fig = plt.figure()
ax = plt.axes()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
for i in range(particles):
    ax.plot(states[i][:, 0], states[i][:, 1])
    ax.scatter(states[i][0][0], states[i][0][1])

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