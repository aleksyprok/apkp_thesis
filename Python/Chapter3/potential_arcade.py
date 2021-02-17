import numpy as np
import matplotlib.pyplot as plt

B0 = 1
Lambda_B = 1

x_min = -np.pi * Lambda_B
x_max =  np.pi * Lambda_B
nx = 512
x = np.linspace(x_min, x_max, nx)

x_min0 = 0
x_max0 = np.pi * Lambda_B / 2
nx0 = 512
x0 = np.linspace(x_min0, x_max0, nx0)

z_min = 0
z_max = 2 * np.pi * Lambda_B
nz = 512
z = np.linspace(z_min, z_max, nz)

X, Z = np.meshgrid(x, z)

A = B0 * Lambda_B * np.cos(X / Lambda_B) * np.exp(-Z / Lambda_B)
a = x0 * Lambda_B * np.cos(x0 / Lambda_B) \
  / (Lambda_B * np.cos(x0 / Lambda_B) + x0 * np.sin(x0 / Lambda_B))


fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = 0.5 * fig_size[1]
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

ax = fig.add_subplot(121)
ax.contour(X / (np.pi * Lambda_B), Z / (np.pi * Lambda_B), A / (B0 * Lambda_B), \
			levels = 20, \
			linestyles = 'solid', \
			colors = 'tab:blue')
ax.set_xlabel(r'$x\, /\, (\pi \Lambda_B)$')
ax.set_ylabel(r'$z\, /\, (\pi \Lambda_B)$')
ax.set_title('Magnetic field lines')

ax = fig.add_subplot(122)
ax.plot(x0 / (np.pi * Lambda_B), a / Lambda_B)
ax.set_xlabel(r'$x_0\, /\, (\pi \Lambda_B)$')
ax.set_title(r'$a(x_0)\, /\, \Lambda_B $')

fig.savefig('temp_figures/potential_arcade.pdf', bbox_inches = 'tight')

plt.show(block = False)