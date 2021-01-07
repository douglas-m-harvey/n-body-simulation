import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as c

dt = 0.001
steps = 3000

G = 1

# format: x, y, z, vx, vy, vz, m
par = [[0, 0, 0, 0, 0, 0, 1], [0.5, 0, 0, 0, 1.63, 0, 1]]

state = np.zeros((len(par), steps + 1, len(par[0])))

for i in range(len(par)):
    state[i][0] = par[i]

def distance(p1_vec, p2_vec):
    X, Y, Z = p1_vec[0:3]
    x, y, z = p2_vec[0:3]
    return ((X - x)**2 + (Y - y)**2 + (Z - z)**2)**(1/2)

def acceleration(p1_vec, p2_vec):
    global G
    M = p1_vec[6]
    X, Y, Z = p1_vec[0:3]
    x, y, z = p2_vec[0:3]
    acc = lambda p1, p2: G*M*(p1 - p2)/distance(p1_vec, p2_vec)**3
    return [acc(X, x), acc(Y, y), acc(Z, z)]

def velocity(p_vec, a_vec, dt):
    vx, vy, vz = p_vec[3:6]
    ax, ay, az = a_vec
    vel = lambda v, a: v + a*dt
    return [vel(vx, ax), vel(vy, ay), vel(vz, az)]
    
def position(p_vec, v_vec, dt):
    x, y, z = p_vec[0:3]
    vx, vy, vz = v_vec
    pos = lambda p, v: p + v*dt
    return [pos(x, vx), pos(y, vy), pos(z, vz)]

for i in range(steps):
    
    r = distance(state[0][i], state[1][i])
    acc1 = acceleration(state[1][i], state[0][i]) # acc on 1
    acc2 = acceleration(state[0][i], state[1][i]) # acc on 2
    
    vel1 = velocity(state[0][i], acc1, dt)
    vel2 = velocity(state[1][i], acc2, dt)
    
    pos1 = position(state[0][i], vel1, dt)
    pos2 = position(state[1][i], vel2, dt)
    
    state1 = np.array([pos1 + vel1 + [state[0][i][6]]])
    state2 = np.array([pos2 + vel2 + [state[0][i][6]]])
    # state1 = np.array([0, 0, 0, 0, 0, 0, 1]) # use th1s to keep p1 at the origin
    
    state[0][i + 1] = state1 # i = 0 to start so this tries to write over first rows
    state[1][i + 1] = state2 # find a better solution to this problem

fig = plt.figure()
ax = plt.axes()
ax.plot(state[0][:, 0], state[0][:, 1])
ax.plot(state[1][:, 0], state[1][:, 1])

# fig = plt.figure()
# ax = plt.axes(projection="3d")
# ax.plot3D(state[0][:, 0], state[0][:, 1], state[0][:, 2])
# ax.plot3D(state[1][:, 0], state[1][:, 1], state[1][:, 2])