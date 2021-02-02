import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

def Lph(omega):
	return (6 * omega * a0** 2 / eta) ** (1/3) / kz(omega)

def kz(omega):
	return omega / vA0

def A_infty(omega, z):
	A_infty = omega - omega + z - z + 0j
	for k in range(Ns):
		zk = (-1) ** k * (z - lz) + (2 * k + 1) * lz + 0j
		A_infty += (-1) ** k * np.exp(-(zk / Lph(omega)) ** 3 - 1j * kz(omega) * zk)
	return A_infty

def B_infty(omega, z):
	B_infty = omega - omega + z - z + 0j
	for k in range(Ns):
		zk = (-1) ** k * (z - lz) + (2 * k + 1) * lz + 0j
		B_infty += np.exp(-(zk / Lph(omega)) ** 3 - 1j * kz(omega) * zk)
	return B_infty

def gamma_approx(omega):
	return 2 * vA0 / Lph(omega)

vA0 = 1
a0= 1
eta = 1e-4
Lz = 1
lz = Lz / 2

omega1 = np.pi * vA0 / Lz
d_omega = omega1 / 100
omega_min = 10 * d_omega
omega_max = 10 * omega1
omega = np.arange(omega_min, 10 * omega1 + d_omega / 2, d_omega)
n_omega = omega.size
print('n_omega = ' + str(n_omega))

z_min = 0
z_max = Lz
nz = 256
dz = (z_max - z_min) / (nz - 1)
z = np.linspace(z_min, z_max, nz)

Omega, Z = np.meshgrid(omega,z)

Ns = int(round(2 * Lph(omega_min) / lz))
print('Ns = ' + str(n_omega))

A_infty0 = A_infty(Omega, Z)
print('A_infty finished')
B_infty0 = B_infty(Omega, Z)
print('B_infty finished')

poy_flux = np.real(A_infty0[0,:] * np.conj(B_infty0[0,:]))
avg_engy = trapz(np.absolute(A_infty0) ** 2 + np.absolute(B_infty0) ** 2, x = z, axis = 0)
gamma0 = 2 * vA0 * poy_flux / avg_engy

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = 0.6 * fig_size[1]
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.2)

ax = fig.add_subplot(121)
ax.plot(omega / omega1, gamma0 / omega1, label = r'$\gamma\, /\, \omega_1$')
ax.plot(omega / omega1, gamma_approx(omega) / omega1, label = r'$(2v_{A0} / L_{ph0})\, /\, \omega_1$')
ax.set_xlabel(r'$\omega / \omega_1$')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, bbox_to_anchor=(0.85,1.3))
ax.text(0.1, 0.9, \
	r'$\eta+\nu = $' + '{:1.1e}'.format(eta) + r'$\eta_0$', \
	transform=ax.transAxes)

n_eta = 25
eta_array = np.logspace(-6,-2, n_eta)
err_array = np.zeros(n_eta)
for i in range(n_eta):

	eta = eta_array[i]
	Ns = int(round(10 * Lph(omega1) / lz))

	A_infty_err = A_infty(omega1, z)
	B_infty_err = B_infty(omega1, z)

	poy_flux = np.real(A_infty_err[0] * np.conj(B_infty_err[0]))
	avg_engy = trapz(np.absolute(A_infty_err) ** 2 + np.absolute(B_infty_err) ** 2, x = z)
	gamma0 = 2 * vA0 * poy_flux / avg_engy

	err_array[i] = np.abs(gamma0 - gamma_approx(omega1)) / gamma0

ax = fig.add_subplot(122)
ax.plot(eta_array, err_array)
ax.set_xscale("log")
ax.set_xlabel(r'$(\eta + \nu) / \eta_0$')
ax.set_title(r'$\left|\frac{\gamma - 2v_{A0}/L_{ph0}}{\gamma}\right|_{\omega=\omega_1}$')
ax.text(0.1, 0.1, \
	r'$\omega = \omega_1 = \pi v_{A0} / L_z$' + '\n' + 
	r'$k_{z0} = \pi / L_z$' + '\n' + 
	r'$L_{ph0} = 6\omega a_0^2 / (\eta+\nu)]^{1/3}/k_{z0}$' + '\n' + 
	r'$a_0 = $' + '{:1.2f}'.format(a0) + r'$L_z$' +  '\n' +  
	r'$\eta_0 = v_{A0} L_z$', \
	transform=ax.transAxes)


fig.savefig('temp_figures/closed_loop_gamma.png', bbox_inches = 'tight', dpi=150)

plt.show(block = False)