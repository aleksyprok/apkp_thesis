import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.sparse import diags
import time

def vA(x):
	return vA0 * (1 + x / a)

def vAx(x):
	return vA0 / a

def dZmdz(z, Zm):

	Zm_xx = np.roll(Zm, -1) - 2 * Zm + np.roll(Zm, 1)
	Zm_xx[0]  = -2 * Zm[0]  + 2 * Zm[1]
	Zm_xx[-1] =  2 * Zm[-2] - 2 * Zm[-1]
	Zm_xx = Zm_xx / dx ** 2

	Zm_z = (-1j * omega * Zm + nu_p * Zm_xx) / vA(x)

	return Zm_z

def jac0_fun():
	main_diag = -(1j * omega + 2 * nu_p / dx ** 2) / vA(x)
	upper_diag = nu_p / dx ** 2 / vA(x[0:-1])
	lower_diag = nu_p / dx ** 2 / vA(x[1:])
	upper_diag[0]  = 2 * nu_p / dx ** 2 / vA(x[0] )
	lower_diag[-1] = 2 * nu_p / dx ** 2 / vA(x[-1])
	diagonals = [main_diag, lower_diag, upper_diag]
	return diags(diagonals, [0, -1, 1])

def jacboian(z, Zm):
	return jac0


def enevelope_u(z):
	return f0 * np.exp(-(z / Lph0) ** 3)

def u_ana(x,z):
	kz = omega / vA(x)
	arg = vAx(x) ** 2 * (eta + nu) / (6 * vA(x) ** 5) * omega ** 2
	return f0 * np.exp(-1j * kz * z) * np.exp(-arg * z ** 3)

errtol = 1e-6

vA0 = 1
B0 = 1
f0 = 1
omega = np.pi
a = 1

eta = 0.5e-4
nu = 0.5e-4
nu_p = 0.5 * (eta + nu)

kx = (6 * omega / (eta + nu) / a) ** (1/3)
lx = 10 / kx
x_min = -lx
x_max = lx
nx = 1024
dx = (x_max - x_min) / nx
x = np.linspace(x_min + dx / 2, x_max - dx / 2, nx)

ix0 = nx // 2
x0 = x[ix0]
kz0 = omega / vA0
Lph0 = (6 * omega * a ** 2 / (eta + nu)) ** (1/3) / kz0

z_min = 0
z_max = 2 * Lph0
nz = 1024
dz = (z_max - z_min) / (nz - 1)
z = np.linspace(z_min, z_max, nz)

X, Z = np.meshgrid(x,z)

Zm0 = np.full_like(x, 2 * f0, dtype = complex)
jac0 = jac0_fun()

t0 = time.time()
sol = solve_ivp(dZmdz, [z_min, z_max], Zm0, t_eval = z, method = 'BDF', jac = jacboian, rtol = errtol, atol = errtol)
t1 = time.time()
print("Time for BDF =", t1 - t0)	

Zm = sol.y.T
u =  0.5 * Zm
b = -0.5 * Zm * B0 / vA(X)

ix0 = nx // 2
iz0 = nz // 2
x0 = x[ix0]
z0 = z[iz0]

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = 1.75 * fig_size[1]
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)

ax = fig.add_subplot(321)
ax.plot(z / Lph0, u[:,ix0].real)
ax.plot(z / Lph0, u[:,ix0].imag)
ax.plot(z / Lph0,  enevelope_u(z), 'tab:green')
ax.plot(z / Lph0, -enevelope_u(z), 'tab:green')
ax.set_xlabel(r'$z / L_{ph0}$')
ax.set_title(r'$u(0,z) / f_0$')

ax = fig.add_subplot(322)
ax.plot(x / lx, u[iz0,:].real)
ax.plot(x / lx, u[iz0,:].imag)
ax.plot(x / lx, u_ana(x,z0).real, '+', color = 'tab:blue', markevery = 10)
ax.plot(x / lx, u_ana(x,z0).imag, '+', color = 'tab:orange', markevery = 10)

ax.set_xlabel(r'$x / l_x$')
ax.set_title(r'$u(x,L_{ph0}) / f_0$')

ax = fig.add_subplot(323)
cp = ax.contourf(X / lx, Z / Lph0, u.real, levels=100)
plt.colorbar(cp)
ax.set_xlabel(r'$x / l_x$')
ax.set_ylabel(r'$z / L_{ph0}$')
ax.set_title(r'Re$[u(x,z)] / f_0$')

legend_elements = [Line2D([0], [0], color = 'tab:blue', label = 'Real part'), \
                   Line2D([0], [0], color = 'tab:orange', label = 'Imag part'), \
                   Line2D([0], [0], color = 'tab:green', label = 'Envelope'),]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=legend_elements, bbox_to_anchor=(2.4,-0.3))

ax = fig.add_subplot(324)
cp = ax.contourf(X / lx, Z / Lph0, u.imag, levels=100)
plt.colorbar(cp)
ax.set_xlabel(r'$x / l_x$')
ax.set_ylabel(r'$z / L_{ph0}$')
ax.set_title(r'Im$[u(x,z)] / f_0$')

legend_elements = [Line2D([0], [0], color = 'k', label = 'Numerical'),
                   Line2D([0], [0], marker = '+', lw = 0, color = 'k', label = 'Analytic')]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=legend_elements, bbox_to_anchor=(1.4,-0.3))

ax.text(0, -1.8, \
	r'$k_{z0} = \omega / v_{A0}$' + '\n' + 
	r'$a_0 = $' + '{:1.2f}'.format(a) + r'$\,\pi / k_{z0}$' '\n' + 
	r'$L_{ph0} = [6\omega a_0^2 / (\eta+\nu)]^{1/3}/k_{z0}$' + '\n' + 
	r'$k_{x0}^* = \{6\omega / [(\eta+\nu)a_0]\}^{1/3}$' + '\n' + 
	r'$l_x = $' + '{:1.1f}'.format(lx * kx) + r'$ / k_{x0}^*$' + '\n' + 
	r'$\eta_0 = \pi v_{A0} / k_{z0}$' + '\n' + 
	r'$\eta + \nu = $' + '{:1.1e}'.format(eta + nu) + r'$\,\eta_0$' + '\n' + 
	r'$n_x \times n_z = $' + '{:d}'.format(nx) + r'$ \times $' + '{:d}'.format(nz), \
	transform=ax.transAxes)


n_eta = 25
eta_nu_array = np.logspace(-7,-2, n_eta)
eta_array = 0.5 * eta_nu_array
nu_array  = 0.5 * eta_nu_array
err_array = np.zeros(n_eta)

for i in range(n_eta):

	print(i + 1, ' / ', n_eta)

	eta = eta_array[i]
	nu =  nu_array[i]
	nu_p = 0.5 * (eta + nu)

	kx = (6 * omega / (eta + nu) / a) ** (1/3)
	lx = min([100 / kx, 0.5])
	x_min = -lx
	x_max = lx
	dx = (x_max - x_min) / nx
	x = np.linspace(x_min + dx / 2, x_max - dx / 2, nx)

	ix0 = nx // 2
	x0 = x[ix0]
	kz0 = omega / vA0
	Lph0 = (6 * omega * a ** 2 / (eta + nu)) ** (1/3) / kz0

	Zm0 = np.full_like(x, 2 * f0, dtype = complex)
	jac0 = jac0_fun()
	sol = solve_ivp(dZmdz, [0, Lph0], Zm0, method = 'BDF', jac = jacboian, rtol = errtol, atol = errtol)
	Zm = sol.y.T
	u =  0.5 * Zm

	err_array[i] = np.abs(u[-1,ix0] - u_ana(x0,Lph0)) / np.abs(u[-1,ix0])

ax = fig.add_subplot(325)
ax.loglog(eta_nu_array, err_array)
ax.set_xlabel(r'$(\eta + \nu) / \eta_0$')
ax.set_title(r'$\frac{|u_{numeric}(0,L_{ph0}) - u_{analytic}(0,L_{ph0})|}{|u_{numeric}(0,L_{ph0})|}$')

fig.savefig('temp_figures/phase_mixing_open_loop.png', bbox_inches = 'tight', dpi=150)

plt.show(block = False)