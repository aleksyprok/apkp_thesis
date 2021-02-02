import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fftpack import diff as psdiff

def vA(x):
	# return vA0 * (1 + a / np.pi * np.arcsin(np.sin(np.pi / Lx * x)))
	return vA0 * (1 + a * np.sin(np.pi / Lx * x))

def vAx(x):
	return np.pi * vA0 * a / Lx * np.cos(np.pi / Lx * x)

def dZdz(z, Z):

	Zp = Z[0:nx]
	Zm = Z[nx:]

	Zp_xx = psdiff(Zp, order = 2, period = 2 * Lx)
	Zm_xx = psdiff(Zm, order = 2, period = 2 * Lx)

	Zp_z = ( 1j * omega * Zp + nu_p * Zp_xx + nu_m * Zm_xx) / vA(x)
	Zm_z = (-1j * omega * Zm + nu_p * Zm_xx + nu_m * Zp_xx) / vA(x)

	return np.concatenate((Zp_z, Zm_z))

def enevelope_u(z):
	return f0 * np.exp(-(z / Lph) ** 3)

def enevelope_b(z):
	return B0 * f0 / vA(x0) * np.exp(-(z / Lph) ** 3)

def u_ana(x,z):
	kz = omega / vA(x)
	arg = vAx(x) ** 2 * (eta + nu) / (6 * vA(x) ** 5) * omega ** 2
	return f0 * np.exp(-1j * kz * z) * np.exp(-arg * z ** 3)

vA0 = 1
B0 = 1
f0 = 1
omega = np.pi
Lz = 2 * np.pi * vA0 / omega
Lx = Lz
a = 0.25

eta = 0.25e-2
nu = 0.75e-2
# eta = 0
# nu = 0
nu_p = 0.5 * (eta + nu)
nu_m = 0.5 * (eta - nu)

x_min = -Lx
x_max = Lx
nx = 128
dx = (x_max - x_min) / nx
x = np.linspace(x_min + dx / 2, x_max - dx / 2, nx)

ix0 = nx // 2
x0 = x[ix0]
Lph = (vAx(x0) ** 2 * (eta + nu) / (6 * vA(x0) ** 5) * omega ** 2) ** (-1/3)

z_min = 0
z_max = 2 * Lph
nz = 256
dz = (z_max - z_min) / (nz - 1)
z = np.linspace(z_min, z_max, nz)

X, Z = np.meshgrid(x,z)

Zp0 = np.full(nx,      0, dtype = complex)
Zm0 = np.full(nx, 2 * f0, dtype = complex)
Z0 = np.concatenate((Zp0, Zm0))

sol = solve_ivp(dZdz, [z_min, z_max], Z0, t_eval = z, method = 'BDF')
Zp = sol.y[0:nx,:].T
Zm = sol.y[nx:,:].T
u = 0.5 * (Zp + Zm)
b = 0.5 * (Zp - Zm) * B0 / vA(X)

############# Create Plots #######################

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = 1.75 * fig_size[1]
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)

ax = fig.add_subplot(321)
ax.plot(z / Lph, np.real(u[:,ix0]))
ax.plot(z / Lph, np.imag(u[:,ix0]))
ax.plot(z / Lph,  enevelope_u(z), 'g')
ax.plot(z / Lph, -enevelope_u(z), 'g')
ax.set_xlabel(r'$z / L_{ph}$')
ax.set_title(r'$u(0,z) / u_0$')

ax = fig.add_subplot(322)
ax.plot(z / Lph, np.real(b[:,ix0]), label = 'Real part')
ax.plot(z / Lph, np.imag(b[:,ix0]), label = 'Imaginary part')
ax.plot(z / Lph,  enevelope_b(z), 'g', label = 'Envelope')
handles, labels = ax.get_legend_handles_labels()
ax.plot(z / Lph, -enevelope_b(z), 'g')
lgd = ax.legend(handles, labels, bbox_to_anchor=(0.8,-1.8))
ax.set_xlabel(r'$z / L_{ph}$')
ax.set_title(r'$b(0,z) / b_0$')

ax = fig.add_subplot(323)
cp = ax.contourf(X / Lx, Z / Lph, np.real(u), levels=100)
plt.colorbar(cp)
ax.set_xlabel(r'$x / L_x$')
ax.set_ylabel(r'$z / L_{ph}$')
ax.set_title(r'Re$[u(x,z)] / u_0$')

ax = fig.add_subplot(324)
cp = ax.contourf(X / Lx, Z / Lph, np.real(b), levels=100)
plt.colorbar(cp)
ax.set_xlabel(r'$x / L_x$')
ax.set_title(r'Re$[b(x,z)] / b_0$')
ax.text(-0.1, -1.8, \
	r'$L_x = L_z = 2 \pi v_{A0} / \omega$' + '\n' + 
	r'$L_{ph} = [(dv_A/dx)^2(\eta+\nu)\omega^2 / (6v_A^5)]^{-1/3}_{x=0}$' + '\n' + 
	r'$a = $' + '{:1.2f}'.format(a) + '\n' + 
	r'$\eta = $' + '{:1.1e}'.format(eta) + r'$\,\eta_0$' + '\n' + 
	r'$\nu = $' + '{:1.1e}'.format(nu) + r'$\,\eta_0$' + '\n' + 
	r'$u_0 = f_0$' + '\n' + 
	r'$b_0 = B_0 f_0 / v_{A0}$' + '\n' + 
	r'$\eta_0 = L_z v_{A0}$' + '\n' + 
	r'$n_x \times n_z = $' + '{:d}'.format(nx) + r'$ \times $' + '{:d}'.format(nz), \
	transform=ax.transAxes)

# n_eta = 10
# eta_nu_array = np.logspace(-4,-1, n_eta)
# eta_array = 0.25 * eta_nu_array
# nu_array  = 0.75 * eta_nu_array
# err_array = np.zeros(n_eta)

# for i in range(n_eta):

# 	print(i)

# 	eta = eta_array[i]
# 	nu =  nu_array[i]
# 	nu_p = 0.5 * (eta + nu)
# 	nu_m = 0.5 * (eta - nu)

# 	Lph = (vAx(x0) ** 2 * (eta + nu) / (6 * vA(x0) ** 5) * omega ** 2) ** (-1/3)

# 	sol = solve_ivp(dZdz, [0, Lph], Z0)
# 	Zp = sol.y[0:nx,:].T
# 	Zm = sol.y[nx:,:].T
# 	u = 0.5 * (Zp + Zm)

# 	err_array[i] = np.abs(u[-1,ix0] - u_ana(x0,Lph)) / np.abs(u[-1,ix0])

# ax = fig.add_subplot(325)
# ax.loglog(eta_nu_array, err_array)
# ax.set_xlabel(r'$(\eta + \nu) / \eta_0$')
# ax.set_title(r'$\frac{|u_{numeric}(0,L_{ph}) - u_{analytic}(0,L_{ph})|}{|u_{numeric}(0,L_{ph})|}$')

fig.savefig('temp_figures/phase_mixing_open_loop.png', bbox_inches = 'tight', dpi=150)

plt.show(block = False) 
