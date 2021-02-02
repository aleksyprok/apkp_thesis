import numpy as np
import matplotlib.pyplot as plt

def vA(x):
	return vA0 * (1 + x / Lx)

def u_ana(x,z,t):
	return u0 * np.heaviside(t - z / vA(x), 1) * np.sin(omega * (t - z / vA(x)))

Lx = 1
vA0 = 1
u0 = 1
omega = np.pi
Lz = 2 * np.pi * vA0 / omega
T = 2 * np.pi / omega

x_min = 0
x_max = Lx
nx = 256
dx = (x_max - x_min) / (nx - 1)
x = np.linspace(x_min, x_max, nx)

z_min = 0
z_max = 10 * Lz
nz = 512
dz = (z_max - z_min) / (nz - 1)
z = np.linspace(z_min, z_max, nz)

t_min = 0
t_max = 10
nt = 512
dt = (t_max - t_min) / (nt - 1)
t = np.linspace(t_min, t_max, nt)

X, Z = np.meshgrid(x,z)

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = 2 * fig_size[1]
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.35)

t0 = 0.25 * T
z0 = vA0 * (t0 - 0.25 * T)
ax = fig.add_subplot(421)
ax.plot(x / Lx, u_ana(x,z0,t0))
ax.set_title(r'$u(x,z_0,t) / u_0$')
ax.text(0.975, 1.05, \
	r'$t = $' + '{:1.2f}'.format(t0 / T) + r'$\,T$', \
	# r'$z_0 = vA0 (t - T / 4)$', \
	fontsize = 12, \
	transform=ax.transAxes)
ax.text(0.2, 0.8, \
	r'$z_0 = v_{A0} (t - T / 4)$', \
	fontsize = 12, \
	transform=ax.transAxes)
ax = fig.add_subplot(422)
cp = ax.contourf(X / Lx, Z / Lz, np.real(u_ana(X,Z,t0)), levels=100)
plt.colorbar(cp)
ax.set_ylabel(r'$z / L_z$')
ax.set_title(r'$u(x,z,t) / u_0$')

t0 = 2.75 * T
z0 = vA0 * (t0 - 0.25 * T)
ax = fig.add_subplot(423)
ax.plot(x / Lx, u_ana(x,z0,t0))
ax.text(0.975, 1.05, \
	r'$t = $' + '{:1.2f}'.format(t0 / T) + r'$\,T$', \
	fontsize = 12, \
	transform=ax.transAxes)
ax = fig.add_subplot(424)
cp = ax.contourf(X / Lx, Z / Lz, np.real(u_ana(X,Z,t0)), levels=100)
plt.colorbar(cp)
ax.set_ylabel(r'$z / L_z$')

t0 = 5.25 * T
z0 = vA0 * (t0 - 0.25 * T)
ax = fig.add_subplot(425)
ax.plot(x / Lx, u_ana(x,z0,t0))
ax.text(0.975, 1.05, \
	r'$t = $' + '{:1.2f}'.format(t0 / T) + r'$\,T$', \
	fontsize = 12, \
	transform=ax.transAxes)
ax = fig.add_subplot(426)
cp = ax.contourf(X / Lx, Z / Lz, np.real(u_ana(X,Z,t0)), levels=100)
plt.colorbar(cp)
ax.set_ylabel(r'$z / L_z$')

t0 = 7.75 * T
z0 = vA0 * (t0 - 0.25 * T)
ax = fig.add_subplot(427)
ax.plot(x / Lx, u_ana(x,z0,t0))
ax.set_xlabel(r'$x / L_x$')
ax.text(0.975, 1.05, \
	r'$t = $' + '{:1.2f}'.format(t0 / T) + r'$\,T$', \
	fontsize = 12, \
	transform=ax.transAxes)
ax = fig.add_subplot(428)
cp = ax.contourf(X / Lx, Z / Lz, np.real(u_ana(X,Z,t0)), levels=100)
plt.colorbar(cp)
ax.set_xlabel(r'$x / L_x$')
ax.set_ylabel(r'$z / L_z$')

fig.savefig('temp_figures/phase_mixing.png', bbox_inches = 'tight', dpi=150)

plt.show(block = False)