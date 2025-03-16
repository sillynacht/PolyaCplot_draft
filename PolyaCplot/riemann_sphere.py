import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

phi = np.linspace(0, np.pi, 50)
theta = np.linspace(0, 2*np.pi, 50)
phi, theta = np.meshgrid(phi, theta)

x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

ax.plot_surface(x, y, z, color='lightblue', alpha=0.5, rstride=4, cstride=4, edgecolor='none')

n_lines = 10

for phi_const in np.linspace(0, np.pi, n_lines):
    theta_line = np.linspace(0, 2*np.pi, 100)
    x_line = np.sin(phi_const) * np.cos(theta_line)
    y_line = np.sin(phi_const) * np.sin(theta_line)
    z_line = np.cos(phi_const)
    ax.plot(x_line, y_line, z_line, color='k', lw=1)

for theta_const in np.linspace(0, 2*np.pi, n_lines):
    phi_line = np.linspace(0, np.pi, 100)
    x_line = np.sin(phi_line) * np.cos(theta_const)
    y_line = np.sin(phi_line) * np.sin(theta_const)
    z_line = np.cos(phi_line)
    ax.plot(x_line, y_line, z_line, color='k', lw=1)

ax.set_box_aspect([1,1,1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
