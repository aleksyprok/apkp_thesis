import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def vA(x):
	return vA0 * (1 + x / a0)

def dvA(x):
	return vA0 / a0

def a(x):
	return vA(x) / dvA(x)

def uxn_ana(x):
	return -1j * ky * uy_n0 * xi_n * np.log((x - 1j * xi_n) / xi_n)

def duxn_ana(x):
	return -1j * ky * uy_n0 * xi_n / (x - 1j * xi_n)

def uyn_ana(x):
	return uy_n0 * xi_n / (x - 1j * xi_n)

def bzn_ana(x):
	return 2 * uy_n0 * xi_n * omega / (ky * vA(x) ** 2 * a(x))

def Sxn_ana(x):
	return -0.5 * uy_n0 ** 2 * xi_n ** 2 * omega / (vA(x) ** 2 * a(x)) * \
			(np.arctan2(xi_n, x) - np.arctan2(-xi_n, x))

def Ln(x):
	return omega ** 2 / vA(x) ** 2 - kzn ** 2

def dLn(x):
	return -2 * omega ** 2 / (vA(x) ** 2 * a(x))

def uxn_eqn(x, U):
	uxn = U[0]
	vxn = U[1]
	duxn = vxn
	dvxn = ky ** 2 * dLn(x) / ((Ln(x) - ky ** 2) * Ln(x)) * vxn \
		- (Ln(x) - ky ** 2) * uxn
	return np.array([duxn, dvxn])

Lz = 1
vA0 = 1
a0 = 1
uy_n0 = 1
n = 1
kzn = n * np.pi / Lz
ky = 2 * kzn

errtol = 1e-8

omega_r = np.pi * vA0 / Lz
omega_i = 1e-4 * omega_r
omega = omega_r + 1j * omega_i

xi_n = a0 * omega_i / omega_r

nx = 256
lx = 8 * abs(xi_n)
x_min = -lx
x_max =  lx
dx = (x_max - x_min) / (nx - 1)
x = np.linspace(x_min, x_max, nx)

uxn0 = uxn_ana(x_min)
vxn0 = duxn_ana(x_min)
U0 = np.array([uxn0, vxn0])

sol = solve_ivp(uxn_eqn, [x_min, x_max], U0, t_eval = x, method = 'RK45', \
				rtol = errtol, atol = errtol)

uxn = sol.y[0,:]
duxn = sol.y[1,:]
uyn = -1j * ky * duxn / (Ln(x) - ky ** 2)
bzn = -(duxn + 1j * ky * uyn) / (1j * omega)
Sxn = 0.25 * (uxn * np.conj(bzn) + np.conj(uxn) * bzn)

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = 1.75 * fig_size[1]
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.45)

ax = fig.add_subplot(321)
ax.plot(x / abs(xi_n), uxn.real, linewidth = 3, label = 'Numerical solution')
ax.plot(x / abs(xi_n), uxn_ana(x).real, label = 'Analytic solution')
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel('$x / |x_{i,1}|$')
ax.set_title(r'Re[$u_{x1} / u_{y10}$]')
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,1.5))

ax = fig.add_subplot(322)
ax.plot(x / abs(xi_n), uxn.imag, linewidth = 3)
ax.plot(x / abs(xi_n), uxn_ana(x).imag)
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel('$x / |x_{i,1}|$')
ax.set_title(r'Im[$u_{x1} / u_{y10}$]')
text1 = ax.text(0.25,1.175, \
	r"$\omega_i / \omega_r = $" + '{:.2e}'.format(omega_i / omega_r) + '\n' + \
	r"$a_0 / L_z = $" + '{:.2f}'.format(a0) + '\n' + \
	r"$k_y / L_z = $" + '{:.2f}'.format(ky / (Lz * np.pi)) + r'$\pi$', \
	transform=ax.transAxes)

ax = fig.add_subplot(323)
ax.plot(x / abs(xi_n), uyn.real, linewidth = 3)
ax.plot(x / abs(xi_n), uyn_ana(x).real)
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel('$x / |x_{i,1}|$')
ax.set_title(r'Re[$u_{y1} / u_{y10}$]')

ax = fig.add_subplot(324)
ax.plot(x / abs(xi_n), uyn.imag, linewidth = 3)
ax.plot(x / abs(xi_n), uyn_ana(x).imag)
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel('$x / |x_{i,1}|$')
ax.set_title(r'Im[$u_{y1} / u_{y10}$]')

Sxn0 = np.pi * uy_n0 ** 2 * xi_n ** 2 * omega_r / (vA0 **2 * a0)
ax = fig.add_subplot(325)
ax.plot(x / abs(xi_n), Sxn.real / Sxn0, linewidth = 3)
ax.plot(x / abs(xi_n), Sxn_ana(x).real / Sxn0)
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlabel('$x / |x_{i,1}|$')
ax.set_title(r'$\langle S_{x1} \rangle / |\langle \Delta S_{x1} \rangle|$')

n_errs = 32
omega_i_min_norm = -7
omega_i_max_norm = -2
omega_i_array = np.logspace(omega_i_min_norm, omega_i_max_norm, n_errs) * omega_r
max_err = np.zeros(n_errs)

for i in range(n_errs):

	omega_i = omega_i_array[i]
	omega = omega_r + 1j * omega_i

	xi_n = a0 * omega_i / omega_r

	nx = 256
	lx = 8 * abs(xi_n)
	x_min = -lx
	x_max =  lx
	dx = (x_max - x_min) / (nx - 1)
	x = np.linspace(x_min, x_max, nx)

	uxn0 = uxn_ana(x_min)
	vxn0 = duxn_ana(x_min)
	U0 = np.array([uxn0, vxn0])

	sol = solve_ivp(uxn_eqn, [x_min, x_max], U0, t_eval = x, method = 'RK45', \
					rtol = errtol, atol = errtol)

	uxn = sol.y[0,:]
	max_err[i] = np.max(np.abs(uxn - uxn_ana(x)) / np.abs(uxn))

ax = fig.add_subplot(326)
ax.loglog(omega_i_array / omega_r, max_err)
ax.set_xlabel('$\omega_i / \omega_r$')
# ax.set_title(r'Max $|u_{x1,num} - u_{x1,ana}|\ \slash\ |u_{x1,num}|$')
ax.set_title(r'Max relative $u_x$ error')

fig.savefig('temp_figures/normal_mode_along_x_alpha=0.pdf', bbox_inches = 'tight')

plt.show(block = False)