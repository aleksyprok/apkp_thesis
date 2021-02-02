import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_bvp

def vA(x):
	return vA0 * (1 + x / a0)

def vAx(x):
	return vA0 / a0

def a(x):
	return vA(x) / vAx(x)


def dYdx(x, Y):

	y = Y[0] + 1j * Y[1]
	v = Y[2] + 1j * Y[3]

	dy = v
	dv = 1j * (omega ** 2 - vA(x) ** 2 * kzn ** 2) * y / (omega * eta) \
	   + 2j * omega * f0 / (n * np.pi * eta)

	return np.array([dy.real, dy.imag, dv.real, dv.imag])

def bcs(Y_a, Y_b):
	return np.array([Y_a[2], Y_a[3], Y_b[2], Y_b[3]])

def kz(x):
	return omega / vA(x)

def Lph(x):
	return (6 * omega * a(x) ** 2 / eta) ** (1/3) / kz(x)

def A_infty(x,z):
	A_infty = x - x + z - z + 0j
	for k in range(Ns):
		if k % 100 == 0: print(k)
		zk = (-1) ** k * (z - lz) + (2 * k + 1) * lz + 0j
		A_infty += (-1) ** k * np.exp(-(zk / Lph(x)) ** 3 - 1j * kz(x) * zk)
	return A_infty

def u_ana(x,z):
	return f0 * A_infty(x,z)

vA0 = 1
B0 = 1
f0 = 1
Lz = 1
lz = Lz / 2
a0 = 1
omega = np.pi * vA0 / Lz
eta = 1e-4
Nh = 10
Ns = int(round(10 * Lph(0) / lz))

kx = (6 * omega / (eta * a0)) ** (1/3)
lx = 10 / kx
x_min = -lx
x_max = lx
nx = 512
dx = (x_max - x_min) / nx
x = np.linspace(x_min + dx / 2, x_max - dx / 2, nx)

z_min = 0
z_max = Lz
nz = 128
dz = (z_max - z_min) / (nz - 1)
z = np.linspace(z_min, z_max, nz)

X, Z = np.meshgrid(x,z)
ix0 = nx // 2
iz0 = nz // 2
x0 = x[ix0]
z0 = z[iz0]

y = np.zeros((nz, nx), dtype = complex)
x_temp = np.linspace(x_min, x_max, 5)
Y = np.zeros((4, x_temp.size))
Y[0, 2] = 1
for n in range(1,Nh+1):
	kzn = n * np.pi / Lz
	sol = solve_bvp(dYdx, bcs, x_temp, Y)
	y += (sol.sol(X)[0] + 1j * sol.sol(X)[1]) * np.sin(kzn * Z)
u = y + f0 * (Lz - Z) / Lz

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = 1.75 * fig_size[1]
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)

ax = fig.add_subplot(321)
ax.plot(z / Lz, u[:,ix0].real)
ax.plot(z / Lz, u[:,ix0].imag)
ax.plot(z / Lz, u_ana(x0,z).real, '+', color = 'tab:blue', markevery = 5)
ax.plot(z / Lz, u_ana(x0,z).imag, '+', color = 'tab:orange', markevery = 5)
ax.set_xlabel(r'$z / L_z$')
ax.set_title(r'$u(0,z) / f_0$')

ax = fig.add_subplot(322)
ax.plot(x / lx, u[iz0,:].real)
ax.plot(x / lx, u[iz0,:].imag)
ax.plot(x / lx, u_ana(x,z0).real, '+', color = 'tab:blue', markevery = 15)
ax.plot(x / lx, u_ana(x,z0).imag, '+', color = 'tab:orange', markevery = 15)
ax.set_xlabel(r'$x / l_x$')
ax.set_title(r'$u(x,l_z) / f_0$')

ax = fig.add_subplot(323)
cp = ax.contourf(X / lx, Z / Lz, u.real, levels=100)
plt.colorbar(cp)
ax.set_xlabel(r'$x / l_x$')
ax.set_ylabel(r'$z / L_z$')
ax.set_title(r'Re$[u(x,z)] / f_0$')

legend_elements = [Line2D([0], [0], color = 'tab:blue', label = 'Real part'), \
                   Line2D([0], [0], color = 'tab:orange', label = 'Imag part')]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=legend_elements, bbox_to_anchor=(2.4,-0.3))

ax = fig.add_subplot(324)
cp = ax.contourf(X / lx, Z / Lz, u.imag, levels=100)
plt.colorbar(cp)
ax.set_xlabel(r'$x / l_x$')
ax.set_ylabel(r'$z / L_z$')
ax.set_title(r'Im$[u(x,z)] / f_0$')

legend_elements = [Line2D([0], [0], color = 'k', label = 'Numerical'),
                   Line2D([0], [0], marker = '+', lw = 0, color = 'k', label = 'Analytic')]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=legend_elements, bbox_to_anchor=(1.4,-0.3))

ax.text(0, -1.7, \
	r'$\omega = \pi v_{A0} / L_z$' '\n' +  
	r'$a_0 = $' + '{:1.2f}'.format(a0) + r'$L_z$' + '\n' +  
	r'$k_{x0}^* = \{6\omega / [(\eta+\nu)a_0]\}^{1/3}$' + '\n' + 
	r'$l_x = $' + '{:1.1f}'.format(lx * kx) + r'$ / k_{x0}^*$' + '\n' + 
	r'$\eta_0 = v_{A0} L_z$' + '\n' + 
	r'$\eta + \nu = $' + '{:1.1e}'.format(eta) + r'$\,\eta_0$' + '\n' + 
	r'$n_x \times n_z = $' + '{:d}'.format(nx) + r'$ \times $' + '{:d}'.format(nz) + '\n' + \
	r'$N_h = $' + '{:d}'.format(Nh), \
	transform=ax.transAxes)

n_eta = 25
eta_array = np.logspace(-6,-2, n_eta)
err_array = np.zeros(n_eta)
for i in range(n_eta):

	print(i)

	eta = eta_array[i]
	Ns = int(round(10 * Lph(0) / lz))

	y = np.zeros((nz, nx), dtype = complex)
	Y = np.zeros((4, x_temp.size))
	Y[0, 2] = 1
	for n in range(1,Nh+1):
		kzn = n * np.pi / Lz
		sol = solve_bvp(dYdx, bcs, x_temp, Y)
		y += (sol.sol(X)[0] + 1j * sol.sol(X)[1]) * np.sin(kzn * Z) 
	u = y + f0 * (Lz - Z) / Lz

	err_array[i] = np.abs(u[iz0,ix0] - u_ana(x0,z0)) / np.abs(u[iz0,ix0])

ax = fig.add_subplot(325)
ax.loglog(eta_array, err_array)
ax.set_xlabel(r'$(\eta + \nu) / \eta_0$')
ax.set_title(r'$\frac{|u_{numeric}(0,l_z) - u_{analytic}(0,l_z)|}{|u_{numeric}(0,l_z)|}$')

fig.savefig('temp_figures/phase_mixing_closed_loop.png', bbox_inches = 'tight', dpi=150)

plt.show(block = False)