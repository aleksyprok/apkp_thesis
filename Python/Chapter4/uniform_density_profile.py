import numpy as np
import matplotlib.pyplot as plt

def ux_ana(z):
	for n in range(3):
		if n == 0:
			ux  = ux0[n] * np.exp(1j * kz[n] * z)
		else:
			ux += ux0[n] * np.exp(1j * kz[n] * z)
	return ux

def u_perp_ana(z):
	for n in range(3):
		if n == 0:
			u_perp  = u_perp0[n] * np.exp(1j * kz[n] * z)
		else:
			u_perp += u_perp0[n] * np.exp(1j * kz[n] * z)
	return u_perp

def bx_ana(z):
	for n in range(3):
		if n == 0:
			bx  = bx0[n] * np.exp(1j * kz[n] * z)
		else:
			bx += bx0[n] * np.exp(1j * kz[n] * z)
	return bx

def b_perp_ana(z):
	for n in range(3):
		if n == 0:
			b_perp  = b_perp0[n] * np.exp(1j * kz[n] * z)
		else:
			b_perp += b_perp0[n] * np.exp(1j * kz[n] * z)
	return b_perp

def b_par_ana(z):
	for n in range(3):
		if n == 0:
			b_par  = b_par0[n] * np.exp(1j * kz[n] * z)
		else:
			b_par += b_par0[n] * np.exp(1j * kz[n] * z)
	return b_par

Lz = 1
nz = 1024
z_min = 0
z_max = Lz
dz = (z_max - z_min) / (nz - 1)
z = np.linspace(z_min, z_max, nz)

vA0 = 1
alpha = 0.25 * np.pi
omega = np.pi * np.cos(alpha) * vA0 / Lz
kx = 50 * omega / vA0
ky = 0.5 * omega / vA0
u0 = 1

kz0 = omega / vA0 / np.cos(alpha)

kz = np.array([
				 kz0 - ky * np.tan(alpha), \
				-kz0 - ky * np.tan(alpha), \
				 1j * np.sqrt(kx ** 2 + ky ** 2 - omega ** 2 / vA0 ** 2 + 0j), \
				])

nabla_par0  = 1j * (ky * np.sin(alpha) + kz * np.cos(alpha))
nabla_perp0 = 1j * (ky * np.cos(alpha) - kz * np.sin(alpha))
L0 = nabla_par0 ** 2 + omega ** 2 / vA0 ** 2
ux_hat = -1j * kx * nabla_perp0 / (L0 - kx ** 2)

u_perp0 = np.zeros(3, dtype=np.complex)
u_perp0[0] =  u0
u_perp0[1] = -u0 * (ux_hat[0] - ux_hat[2]) / (ux_hat[1] - ux_hat[2])
u_perp0[2] =  u0 * (ux_hat[0] - ux_hat[1]) / (ux_hat[1] - ux_hat[2])

ux0 = ux_hat * u_perp0
bx0 = nabla_perp0 * ux0 / (1j * omega)
b_perp0 = nabla_perp0 * u_perp0 / (1j * omega)
b_par0 = -(1j * kx * ux0 + nabla_perp0 * u_perp0) / (1j * omega)

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = 1.75 * fig_size[1]
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)

ax = fig.add_subplot(321)
ax.plot(z, ux_ana(z).real)
ax.plot(z, ux_ana(z).imag)
ax.set_title(r'$u_x(0,0,z,0) / u_0$')

ax = fig.add_subplot(322)
ax.plot(z, u_perp_ana(z).real)
ax.plot(z, u_perp_ana(z).imag)
ax.set_title(r'$u_\perp(0,0,z,0) / u_0$')

ax = fig.add_subplot(323)
ax.plot(z, bx_ana(z).real)
ax.plot(z, bx_ana(z).imag)
ax.set_title(r'$\hat{b}_x(0,0,z,0)$')

ax = fig.add_subplot(324)
ax.plot(z, b_perp_ana(z).real)
ax.plot(z, b_perp_ana(z).imag)
ax.set_title(r'$\hat{b}_\perp(0,0,z,0)$')
ax.set_xlabel(r'$z\,/\,L_z$')

ax = fig.add_subplot(325)
ax.plot(z, b_par_ana(z).real, label = 'Real part')
ax.plot(z, b_par_ana(z).imag, label = 'Imag part')
ax.set_title(r'$\hat{b}_{||}(0,0,z,0)$')
ax.set_xlabel(r'$z\,/\,L_z$')
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.7,0.9))
ax.text(1.45, 0.1, \
	r"$\omega = \pi v_{A0} \cos\alpha / L_z$" + '\n' + \
	r"$\alpha = $" + "{:02.3f}".format(alpha / np.pi) + r"$\pi$" + '\n' + \
	r"$k_x$ = " + "{:02.1f}".format(kx / omega) + r"$\omega / v_{A0}$" + '\n' + \
	r"$k_y$ = " + "{:02.1f}".format(ky / omega) + r"$\omega / v_{A0}$",
	transform=ax.transAxes)

fig.savefig('temp_figures/uniform_density_profile.png', bbox_inches = 'tight', dpi=150)

plt.show(block = False)