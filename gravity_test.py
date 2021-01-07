import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as c

dt = 0.01

m1 = 1
m2 = 1
p1 = np.array([[0, 0, 0]])
v1 = np.array([[0, 0, 0]])
p2 = np.array([[0.5, 0, 0]])
v2 = np.array([[-1, 1.63, 0]])

rad = []

for i in range(10000):
    X, Y, Z = p1[0]
    VX, VY, VZ = v1[0]
    x, y, z = p2[i]
    vx, vy, vz = v2[i]
    r = ((x - X)**2 + (y - Y)**2 + (z - Z)**2)**(1/2)
    ax = -x/r**3
    ay = -y/r**3
    az = -z/r**3
    vx += ax*dt
    vy += ay*dt
    vz += az*dt
    x += vx*dt
    y += vy*dt
    z += vz*dt
    p2 = np.append(p2, np.array([[x, y, z]]), axis = 0)
    v2 = np.append(v2, np.array([[vx, vy, vz]]), axis = 0)
    rad.append(r)

a = (max(rad) + min(rad))/2 # semimajor axis
T = 2*c.pi*a**(3/2) # orbital period (4.039)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot3D(p2[:, 0], p2[:, 1], p2[:, 2])

# from celluloid import Camera
# fig = plt.figure()
# camera = Camera(fig)
# for i in range(len(pos_x)):
#     plt.scatter(pos_x[i], pos_y[i], color = "k")
#     plt.scatter(0, 0)
#     camera.snap()
# animation = camera.animate(interval = 1/60*(10**-3))
# animation.save("celluloid_minimal.gif", writer = "pillow")