import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fftpack import diff as psdiff

def dZdz(z, Z):

	Zp = Z[0]
	Zm = Z[1]

	Zp_z = ( 1j * omega * Zp + nu_p * Zp + nu_m * Zm) / vA0
	Zm_z = (-1j * omega * Zm - nu_p * Zm - nu_m * Zp) / vA0

	return np.array([Zp_z, Zm_z])

Lx = 1
Lz = 100
vA0 = 1
omega = np.pi
a = 0.5

eta = 1e-3
nu = 1e-3
nu_p = 0.5 * (eta + nu)
nu_m = 0.5 * (eta - nu)

z_min = 0
z_max = Lz
nz = 256
dz = (z_max - z_min) / (nz - 1)
z = np.linspace(z_min, z_max, nz)

Zp0 = 0 + 0j
Zm0 = 1 + 0j
Z0 = np.array([Zp0, Zm0])

############# Create Plots #######################

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size = 2 * fig_size
fig.set_size_inches(fig_size)

eta = 1
nu = 1
nu_p = 0.5 * (eta + nu)
nu_m = 0.5 * (eta - nu)
sol = solve_ivp(dZdz, [z_min, z_max], Z0, t_eval = z)
Zp = sol.y[0,:]
Zm = sol.y[1,:]

ax = fig.add_subplot(221)
ax.plot(z, np.real(Zp), label = 'Real part')
ax.plot(z, np.imag(Zp), label = 'Imaginary part')
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, bbox_to_anchor=(1.3,1.3))
ax.text(0.9, 1.05, \
	r'$\eta = $' + '{:1.1e}'.format(eta) + r', $\nu = $' + '{:1.1e}'.format(nu), \
	fontsize = 12, \
	transform=ax.transAxes)
ax.set_xlabel(r'$z$')
ax.set_title(r'$\mathcal{Z}^+(0,z)$')

ax = fig.add_subplot(222)
ax.plot(z, np.real(Zm))
ax.plot(z, np.imag(Zm))
ax.set_xlabel(r'$z$')
ax.set_title(r'$\mathcal{Z}^-(0,z)$')

eta = 1
nu = 0
nu_p = 0.5 * (eta + nu)
nu_m = 0.5 * (eta - nu)
sol = solve_ivp(dZdz, [z_min, z_max], Z0, t_eval = z)
Zp = sol.y[0,:]
Zm = sol.y[1,:]

ax = fig.add_subplot(223)
ax.plot(z, np.real(Zp))
ax.plot(z, np.imag(Zp))
ax.text(0.9, 1.075, \
	r'$\eta = $' + '{:1.1e}'.format(eta) + r', $\nu = $' + '{:1.1e}'.format(nu), \
	fontsize = 12, \
	transform=ax.transAxes)
ax.set_xlabel(r'$z$')
ax.set_title(r'$\mathcal{Z}^+(0,z)$')

ax = fig.add_subplot(224)
ax.plot(z, np.real(Zm))
ax.plot(z, np.imag(Zm))
ax.set_xlabel(r'$z$')
ax.set_title(r'$\mathcal{Z}^-(0,z)$')

plt.show(block = False)