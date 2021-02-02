import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import trapz
from scipy.special import zeta

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

def harmonic_number(N, p):
	if p == 1:
		hn = np.log(N) + np.euler_gamma
	else:
		hn = N ** (-p) * (N / (1 - p) + 0.5) + zeta(p)
	return hn

vA0 = 1
a0= 1
eta = 1e-6
Lz = 1
lz = Lz / 2
N = 100

omega1 = np.pi * vA0 / Lz
d_omega = omega1
omega_min = omega1
omega_max = N * omega1
omega = np.linspace(omega_min, omega_max, N)
n_array = np.arange(1,N+1, dtype = float)

z_min = 0
z_max = Lz
nz = 256
dz = (z_max - z_min) / (nz - 1)
z = np.linspace(z_min, z_max, nz)

Omega, Z = np.meshgrid(omega,z)

Ns = int(round(2 * Lph(omega_min) / lz))
print('Ns = ' + str(N))

A_infty0 = A_infty(Omega, Z)
print('A_infty finished')
B_infty0 = B_infty(Omega, Z)
print('B_infty finished')

poy_flux = np.real(A_infty0[0,:] * np.conj(B_infty0[0,:]))
avg_engy = trapz(np.absolute(A_infty0) ** 2 + np.absolute(B_infty0) ** 2, x = z, axis = 0)
gamma_n = 2 * vA0 * poy_flux / avg_engy

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = 0.6 * fig_size[1]
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

ax = fig.add_subplot(121)
alpha_array = np.array([0,1,5/3])
clrs = ['tab:blue', 'tab:orange', 'tab:green']

i = -1
for alpha in alpha_array:

	i += 1

	gamma_weighted_avg = np.cumsum(n_array ** (-alpha) * gamma_n) / np.cumsum(n_array ** (-alpha))

	# gamma_avg_approx = gamma_approx(omega1) * np.cumsum(n_array ** (2 / 3 - alpha)) \
	# 					/ np.cumsum(n_array ** (-alpha))

	gamma_avg_approx2 = gamma_approx(omega1) * harmonic_number(n_array, alpha - 2 / 3) \
						/ harmonic_number(n_array, alpha)

	ax.plot(n_array, gamma_weighted_avg / omega1, color = clrs[i])
	ax.plot(n_array, 1.1 * gamma_avg_approx2 / omega1, '+', color = clrs[i], markevery = 4)

legend_elements = [Line2D([0], [0], color = 'k', label = 'Exact'),
                   Line2D([0], [0], marker = '+', lw = 0, color = 'k', label = 'Approximate')]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=legend_elements)
ax.set_title(r'$\gamma / \omega_1$')
ax.set_xlabel(r'$N$ (# of harmonics)')

ax.text(0.05, 0.7, \
	r'$\eta+\nu = $' + '{:1.0e}'.format(eta) + r'$\eta_0$', \
	transform=ax.transAxes)

ax.text(0.175, 0.5, \
	r'$\alpha = 0$', \
	color = 'tab:blue', \
	transform=ax.transAxes)
ax.text(0.6, 0.45, \
	r'$\alpha = 1$', \
	color = 'tab:orange', \
	transform=ax.transAxes)
ax.text(0.7, 0.225, \
	r'$\alpha = 5 / 3$', \
	color = 'tab:green', \
	transform=ax.transAxes)

ax = fig.add_subplot(122)
n_p = 1024

p_max = -2
p_min = 0.99
dp = (p_max - p_min) / (n_p - 1)
p = np.linspace(p_min, p_max, n_p)
ax.plot(p, zeta(p), color = 'tab:blue')

p_max = 1.01
p_min = 4
dp = (p_max - p_min) / (n_p - 1)
p = np.linspace(p_min, p_max, n_p)
ax.plot(p, zeta(p), color = 'tab:blue')

ax.plot([1,1], [-3,3], ':', color = 'tab:blue')

ax.set_ylim(-3,3)
ax.xaxis.grid()
ax.yaxis.grid()
ax.axes.set_aspect('equal')
ax.set_title(r'$\zeta(p)$')
ax.set_xlabel(r'$p$')

fig.savefig('temp_figures/multiple_harmonics.png', bbox_inches = 'tight', dpi=150)

plt.show(block = False)