import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fftpack import diff as psdiff

def vA(x):
	# return vA0 * (1 + a / np.pi * np.arcsin(np.sin(np.pi / Lx * x)))
	return vA0 * (1 + a * np.sin(np.pi / Lx * x))

def dZdz(z, Z):

	Zp = Z[0:nx]
	Zm = Z[nx:]

	Zp_xx = psdiff(Zp, order = 2, period = 2 * Lx)
	Zm_xx = psdiff(Zm, order = 2, period = 2 * Lx)

	# Zp_z = ( 1j * omega * Zp + nu_p * Zp_xx + nu_m * Zm_xx) / vA(x)
	# Zm_z = (-1j * omega * Zm + nu_p * Zm_xx + nu_m * Zp_xx) / vA(x)
	Zp_z = Zp_z - Zp_z
	Zm_z = (-1j * omega * Zm + nu_p * Zm_xx) / vA(x)


	return np.concatenate((Zp_z, Zm_z))

Lx = 1
vA0 = 1
f0 = 1
omega = np.pi
Lz = 2 * np.pi * vA0 / omega
a = 0.5

eta = 1e-3
nu = 1e-3
nu_p = 0.5 * (eta + nu)
nu_m = 0.5 * (eta - nu)

x_min = -Lx
x_max = Lx
nx = 1024
dx = (x_max - x_min) / nx
x = np.linspace(x_min + dx / 2, x_max - dx / 2, nx)

z_min = 0
z_max = 100 * Lz
nz = 2048
dz = (z_max - z_min) / (nz - 1)
z = np.linspace(z_min, z_max, nz)

Zp0 = np.full(nx,      0, dtype = complex)
Zm0 = np.full(nx, 2 * f0, dtype = complex)
Z0 = np.concatenate((Zp0, Zm0))

############# Create Plots #######################

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size = 2 * fig_size
fig.set_size_inches(fig_size)

eta = 1e-6
nu = 1e-6
nu_p = 0.5 * (eta + nu)
nu_m = 0.5 * (eta - nu)
sol = solve_ivp(dZdz, [z_min, z_max], Z0, t_eval = z, rtol=1e-5, atol=1e-10)
Zp = sol.y[0:nx,:].T
Zm = sol.y[nx:,:].T

ax = fig.add_subplot(221)
ax.plot(z, np.real(Zp[:,nx//2]), label = 'Real part')
ax.plot(z, np.imag(Zp[:,nx//2]), label = 'Imaginary part')
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, bbox_to_anchor=(1.3,1.3))
ax.text(0.9, 1.05, \
	r'$\eta = $' + '{:1.1e}'.format(eta) + r', $\nu = $' + '{:1.1e}'.format(nu), \
	fontsize = 12, \
	transform=ax.transAxes)
ax.set_xlabel(r'$z$')
ax.set_title(r'$\mathcal{Z}^+(0,z)$')

ax = fig.add_subplot(222)
ax.plot(z, np.real(Zm[:,nx//2]))
ax.plot(z, np.imag(Zm[:,nx//2]))
ax.set_xlabel(r'$z$')
ax.set_title(r'$\mathcal{Z}^-(0,z)$')

eta = 1e-6
nu = 2e-6
nu_p = 0.5 * (eta + nu)
nu_m = 0.5 * (eta - nu)
sol = solve_ivp(dZdz, [z_min, z_max], Z0, t_eval = z, rtol=1e-5, atol=1e-10)
Zp = sol.y[0:nx,:].T
Zm = sol.y[nx:,:].T

ax = fig.add_subplot(223)
ax.plot(z, np.real(Zp[:,nx//2]))
ax.plot(z, np.imag(Zp[:,nx//2]))
ax.text(0.9, 1.075, \
	r'$\eta = $' + '{:1.1e}'.format(eta) + r', $\nu = $' + '{:1.1e}'.format(nu), \
	fontsize = 12, \
	transform=ax.transAxes)
ax.set_xlabel(r'$z$')
ax.set_title(r'$\mathcal{Z}^+(0,z)$')

ax = fig.add_subplot(224)
ax.plot(z, np.real(Zm[:,nx//2]))
ax.plot(z, np.imag(Zm[:,nx//2]))
ax.set_xlabel(r'$z$')
ax.set_title(r'$\mathcal{Z}^-(0,z)$')

fig.savefig('temp_figures/nu=0_vs_nu=1e-3.png', bbox_inches = 'tight', dpi=150)

plt.show(block = False)