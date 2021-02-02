import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.special import hankel2

def vA(x):
	return vA0 * (1 + x / a0)

def vAx(x):
	return vA0 / a0

def a(x):
	return vA(x) / vAx(x)

def kz(x, omega):
	return omega / vA(x)

def Lph(x, omega):
	return (6 * omega * a(x) ** 2 / eta) ** (1/3) / kz(x, omega)

def A_infty(x, z, omega, R):
	A_infty = omega - omega + z - z + x - x + 0j
	for k in range(Ns):
		zk = (-1) ** k * (z - lz) + (2 * k + 1) * lz + 0j
		A_infty += (-1) ** k * R ** k * np.exp(-(zk / Lph(x, omega)) ** 3 - 1j * kz(x, omega) * zk)
	return A_infty

def B_infty(x, z, omega, R):
	B_infty = omega - omega + z - z + x - x + 0j
	for k in range(Ns):
		zk = (-1) ** k * (z - lz) + (2 * k + 1) * lz + 0j
		B_infty += R ** k * np.exp(-(zk / Lph(x, omega)) ** 3 - 1j * kz(x, omega) * zk)
	return B_infty

def u_ana(x, z, omega, R):
	return f0 * A_infty(x, z, omega, R)

def gamma_leaky(x, omega, R):
	return 2 * eta * (omega * Lz / (a(x) * vA(x) * np.abs(np.log(R)))) ** 2

def gamma_resistive(x, omega):
	return 2.2 * vA(x) / Lph(x, omega)

def tau_leakage(x, R):
	return Lz / (vA(x) * np.abs(np.log(R)))

def tau_resistive(x, omega):
	return Lph(x,omega) / vA(x)

vA0 = 1
a0 = 1
eta = 1e-4
Lz = 1
lz = Lz / 2
f0 = 1
R = 0.99
omega = np.pi
x0 = 0
Ns = int(round(10 * Lph(x0, omega) / lz))
# Ns = 300

z_min = 0
z_max = Lz
nz = 128
dz = (z_max - z_min) / (nz - 1)
z = np.linspace(z_min, z_max, nz)

epsilon_min = -3
epsilon_max = -0.1
nR = 128
epsilon = np.logspace(epsilon_min, epsilon_max, nR)
R = 1 - epsilon

RR, Z = np.meshgrid(R,z)

A_infty0 = A_infty(x0, Z, omega, RR)
print('A_infty finished')
B_infty0 = B_infty(x0, Z, omega, RR)
print('B_infty finished')

poy_flux = np.real(A_infty0[0,:] * np.conj(B_infty0[0,:])) - np.real(A_infty0[-1,:] * np.conj(B_infty0[-1,:]))
avg_engy = trapz(np.absolute(A_infty0) ** 2 + np.absolute(B_infty0) ** 2, x = z, axis = 0)
gamma_exact = 2 * vA(x0) * poy_flux / avg_engy
gamma_leaky_approx = gamma_leaky(x0, omega, R)
gamma_resistive_approx = np.full(nR, gamma_resistive(x0,omega))

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = 0.6 * fig_size[1]
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.2)

clr = 'tab:blue'
ax = fig.add_subplot(122)
ax.plot(epsilon, gamma_exact, color = clr, label = r'$\gamma / \omega_1$')
ax.plot(epsilon, gamma_resistive_approx, '--', color = clr, label = r'$2.2v_A / (L_{ph} \omega_1)$')
ylim0 = ax.get_ylim()
ax.plot(epsilon, gamma_leaky_approx, ':', color = clr, label = r'$2(\eta+\nu)\left(\frac{\omega L_z}{avA|\ln R|}\right)^2/\omega_1$')
ax.set_xscale('log')
ax.set_ylim(ylim0)
ax.set_xlabel(r'$-\ln R$')
ax.tick_params(axis='y', labelcolor = clr)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, bbox_to_anchor=(0.5,1.2), loc='center')
ax.text(0.1, 0.1, \
	r'$\omega = \omega_1 = \pi v_{A0} / L_z$' + '\n' + 
	r'$k_{z0} = \pi / L_z$' + '\n' + 
	r'$L_{ph0} = 6\omega a_0^2 / (\eta+\nu)]^{1/3}/k_{z0}$' + '\n' + 
	r'$a_0 = $' + '{:1.2f}'.format(a0) + r'$L_z$' +  '\n' +  
	r'$\eta_0 = v_{A0} L_z$', \
	transform=ax.transAxes)


clr = 'tab:orange'
ax2 = ax.twinx()
ax2.plot(epsilon, tau_leakage(x0,R) / tau_resistive(x0,omega), color = clr)
ax2.set_yscale('log')
ax2.tick_params(axis='y', labelcolor = clr)
ax2.set_ylabel(r'$\tau_{leakage} / \tau_{resistive}$', color = clr)

vA0 = 1e6
a0 = 1e5
eta = 1e1
Lz = 1e8
lz = Lz / 2
h = 150 * 1e3

f_min = -3
f_max =  0
nf = 128
f = np.logspace(f_min, f_max, nf)
omega = 2 * np.pi * f
xi0 = 2 * h * omega / vA0
R = np.abs((hankel2(0,xi0) + 1j * hankel2(1, xi0)) / (hankel2(0,xi0) - 1j * hankel2(1, xi0)))

ax = fig.add_subplot(121)
ax.plot(f, 1 / tau_leakage(x0, R), label = r'$1 / \tau_{leakage}\ (s^{-1})$')
ax.plot(f, 1 / tau_resistive(x0, omega), label = r'$1 / \tau_{resistive}\ (s^{-1})$')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, bbox_to_anchor=(0.5,1.2), loc='center')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$f$ (Hz)')


fig.savefig('temp_figures/leaky_loop.png', bbox_inches = 'tight', dpi=150)

plt.show(block = False)