import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as c

dt = 0.0001

G = 1

# format: x, y, z, vx, vy, vz, m
par1 = np.array([[0, 0, 0, 0, 0, 0, 1]])
par2 = np.array([[0.5, 0, 0, 0, 1.63, 0, 4]])

rad = []

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

for i in range(10000):
    r = distance(par1[i], par2[i])
    acc1 = acceleration(par2[i], par1[i]) # acc on 1
    acc2 = acceleration(par1[i], par2[i]) # acc on 2
    
    vel1 = velocity(par1[i], acc1, dt)
    vel2 = velocity(par2[i], acc2, dt)
    
    pos1 = position(par1[i], vel1, dt)
    pos2 = position(par2[i], vel2, dt)
    
    par1 = np.append(par1, np.array([pos1 + vel1 + [par1[i][6]]]), axis = 0)
    par2 = np.append(par2, np.array([pos2 + vel2 + [par2[i][6]]]), axis = 0)
    rad.append(r)

a = (max(rad) + min(rad))/2 # semimajor axis
T = 2*c.pi*a**(3/2) # orbital period (4.039)

fig = plt.figure()
ax = plt.axes()
ax.plot(par1[:, 0], par1[:, 1])
ax.plot(par2[:, 0], par2[:, 1])

# fig = plt.figure()
# ax = plt.axes(projection="3d")
# ax.plot3D(p1[:, 0], p1[:, 1], p1[:, 2])
# ax.plot3D(p2[:, 0], p2[:, 1], p2[:, 2])

# from celluloid import Camera
# fig = plt.figure()
# camera = Camera(fig)
# for i in range(len(pos_x)):
#     plt.scatter(pos_x[i], pos_y[i], color = "k")
#     plt.scatter(0, 0)
#     camera.snap()
# animation = camera.animate(interval = 1/60*(10**-3))
# animation.save("celluloid_minimal.gif", writer = "pillow")