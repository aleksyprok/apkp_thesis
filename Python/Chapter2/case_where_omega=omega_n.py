import numpy as np
import matplotlib.pyplot as plt
import math

def m(t):
	return np.floor(t * vA0 / Lz)

def m_prime(t):
	return np.floor(t * vA0 / Lz - 0.5)

def u_ana(z,t):
	u = z - z + t - t + 0j
	for k in np.arange(m(t_max)+1):
		theta_k = t - (-1) ** k * z / vA0 - (2 * k + 1) * lz / vA0
		u += (-1) ** k * np.heaviside(theta_k, 1) * np.exp(1j * omega * theta_k)
	u = u * u0
	return u

def b_ana(z,t):
	b = z - z + t - t + 0j
	for k in np.arange(m(t_max)+1):
		theta_k = t - (-1) ** k * z / vA0 - (2 * k + 1) * lz / vA0
		b += np.heaviside(theta_k, 1) * np.exp(1j * omega * theta_k)
	b = -B0 * u0 / vA0 * b
	return b

vA0 = 1
lz = 0.5
Lz = 2 * lz
u0 = 1
B0 = 1
B0 = 1
omega_1 = np.pi * vA0 / Lz

z_min = -lz
z_max = lz
nz = 512
dt = (z_max - z_min) / (nz - 1)
z = np.linspace(z_min, z_max, nz)

t_min = 0
t_max = 10
nt = 512
dt = (t_max - t_min) / (nt - 1)
t = np.linspace(t_min, t_max, nt)

T, Z = np.meshgrid(t,z)

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = 2 * fig_size[1]
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.075, bottom=0.1, right=0.925, top=0.9, wspace=0.4, hspace=0.4)

ax = fig.add_subplot(321)
n = 0
omega = n * omega_1
ax.plot(t, np.real(u_ana(0,t)))
ax.set_xlabel(r'$t / t_0$')
ax.text(1.0, 1.3, \
	r'$\omega = \omega_n$', \
	fontsize = 16, \
	transform=ax.transAxes)
ax.text(1.1, 1.05, \
	r'$n = $' + '{:d}'.format(n), \
	fontsize = 12, \
	transform=ax.transAxes)
ax.set_title(r'Re$[u(0,t)/u_0]$')
ax = fig.add_subplot(322)
cp = ax.contourf(T, Z, np.real(u_ana(Z,T)), levels=100)
plt.colorbar(cp)
ax.set_xlabel(r'$t / t_0$')
ax.set_ylabel(r'$z / L_z$')
ax.set_title(r'Re$[u(z,t)/u_0]$')

ax = fig.add_subplot(323)
n = 1
omega = n * omega_1
ax.plot(t, np.imag(u_ana(0,t)))
ax.set_xlabel(r'$t / t_0$')
ax.text(1.1, 1.05, \
	r'$n = $' + '{:d}'.format(n), \
	fontsize = 12, \
	transform=ax.transAxes)
ax.set_title(r'Im$[u(0,t)/u_0]$')
ax = fig.add_subplot(324)
cp = ax.contourf(T, Z, np.imag(u_ana(Z,T)), levels=100)
plt.colorbar(cp)
ax.set_xlabel(r'$t / t_0$')
ax.set_ylabel(r'$z / L_z$')
ax.set_title(r'Im$[u(z,t)/u_0]$')

ax = fig.add_subplot(325)
n = 2
omega = n * omega_1
ax.plot(t, np.imag(u_ana(0,t)))
ax.set_xlabel(r'$t / t_0$')
ax.text(1.1, 1.05, \
	r'$n = $' + '{:d}'.format(n), \
	fontsize = 12, \
	transform=ax.transAxes)
ax.set_title(r'Im$[u(0,t)/u_0]$')
ax = fig.add_subplot(326)
cp = ax.contourf(T, Z, np.imag(u_ana(Z,T)), levels=100)
plt.colorbar(cp)
ax.set_xlabel(r'$t / t_0$')
ax.set_ylabel(r'$z / L_z$')
ax.set_title(r'Im$[u(z,t)/u_0]$')

fig.savefig('temp_figures/case_where_omega=omega_n_u.png', bbox_inches = 'tight', dpi=150)

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = 2 * fig_size[1]
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.075, bottom=0.1, right=0.925, top=0.9, wspace=0.4, hspace=0.4)

ax = fig.add_subplot(321)
n = 0
omega = n * omega_1
ax.plot(t, np.real(b_ana(0,t)))
ax.set_xlabel(r'$t / t_0$')
ax.text(1.0, 1.3, \
	r'$\omega = \omega_n$', \
	fontsize = 16, \
	transform=ax.transAxes)
ax.text(1.1, 1.05, \
	r'$n = $' + '{:d}'.format(n), \
	fontsize = 12, \
	transform=ax.transAxes)
ax.set_title(r'Re$[b(0,t)/b_0]$')
ax = fig.add_subplot(322)
ax.set_title(r'Re$[b(0,t)/b_0]$')
cp = ax.contourf(T, Z, np.real(b_ana(Z,T)), levels=100)
plt.colorbar(cp)
ax.set_xlabel(r'$t / t_0$')
ax.set_ylabel(r'$z / L_z$')
ax.set_title(r'Re$[b(z,t)/b_0]$')

ax = fig.add_subplot(323)
n = 1
omega = n * omega_1
ax.plot(t, np.imag(b_ana(0,t)))
ax.set_xlabel(r'$t / t_0$')
ax.text(1.1, 1.05, \
	r'$n = $' + '{:d}'.format(n), \
	fontsize = 12, \
	transform=ax.transAxes)
ax.set_title(r'Im$[b(0,t)/b_0]$')
ax = fig.add_subplot(324)
cp = ax.contourf(T, Z, np.imag(b_ana(Z,T)), levels=100)
plt.colorbar(cp)
ax.set_xlabel(r'$t / t_0$')
ax.set_ylabel(r'$z / L_z$')
ax.set_title(r'Im$[b(z,t)/b_0]$')

ax = fig.add_subplot(325)
n = 2
omega = n * omega_1
ax.plot(t, np.imag(b_ana(0,t)))
ax.set_xlabel(r'$t / t_0$')
ax.text(1.1, 1.05, \
	r'$n = $' + '{:d}'.format(n), \
	fontsize = 12, \
	transform=ax.transAxes)
ax.set_title(r'Im$[b(0,t)/b_0]$')
ax = fig.add_subplot(326)
cp = ax.contourf(T, Z, np.imag(b_ana(Z,T)), levels=100)
plt.colorbar(cp)
ax.set_xlabel(r'$t / t_0$')
ax.set_ylabel(r'$z / L_z$')
ax.set_title(r'Im$[b(z,t)/b_0]$')

fig.savefig('temp_figures/case_where_omega=omega_n_b.png', bbox_inches = 'tight', dpi=150)