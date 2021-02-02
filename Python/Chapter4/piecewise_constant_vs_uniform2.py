import numpy as np
import matplotlib.pyplot as plt

def exp_cap(z):
	return np.exp(z * (z <= 100))

def ux_uniform(z):
	for n in range(3):
		if n == 0:
			ux  = ux0[n] * np.exp(1j * kz[n] * z)
		else:
			ux += ux0[n] * np.exp(1j * kz[n] * z)
	return ux

def u_perp_uniform(z):
	for n in range(3):
		if n == 0:
			u_perp  = u_perp0[n] * np.exp(1j * kz[n] * z)
		else:
			u_perp += u_perp0[n] * np.exp(1j * kz[n] * z)
	return u_perp

def bx_uniform(z):
	for n in range(3):
		if n == 0:
			bx  = bx0[n] * np.exp(1j * kz[n] * z)
		else:
			bx += bx0[n] * np.exp(1j * kz[n] * z)
	return bx

def b_perp_uniform(z):
	for n in range(3):
		if n == 0:
			b_perp  = b_perp0[n] * np.exp(1j * kz[n] * z)
		else:
			b_perp += b_perp0[n] * np.exp(1j * kz[n] * z)
	return b_perp

def b_par_uniform(z):
	for n in range(3):
		if n == 0:
			b_par  = b_par0[n] * np.exp(1j * kz[n] * z)
		else:
			b_par += b_par0[n] * np.exp(1j * kz[n] * z)
	return b_par

def ux_piecewise(z):
	for n in range(4):
		if n == 0:
			ux  = ux0_m[n] * exp_cap(1j * m_m[n] * z) * np.heaviside(-z, 0)
			ux += ux0_p[n] * exp_cap(1j * m_p[n] * z) * np.heaviside( z, 1)
		else:
			ux += ux0_m[n] * exp_cap(1j * m_m[n] * z) * np.heaviside(-z, 0)
			ux += ux0_p[n] * exp_cap(1j * m_p[n] * z) * np.heaviside( z, 1)
	return ux

def u_perp_piecewise(z):
	for n in range(4):
		if n == 0:
			u_perp  = u_perp0_m[n] * exp_cap(1j * m_m[n] * z) * np.heaviside(-z, 0)
			u_perp += u_perp0_p[n] * exp_cap(1j * m_p[n] * z) * np.heaviside( z, 1)
		else:
			u_perp += u_perp0_m[n] * exp_cap(1j * m_m[n] * z) * np.heaviside(-z, 0)
			u_perp += u_perp0_p[n] * exp_cap(1j * m_p[n] * z) * np.heaviside( z, 1)
	return u_perp

def bx_piecewise(z):
	for n in range(4):
		if n == 0:
			bx  = bx0_m[n] * exp_cap(1j * m_m[n] * z) * np.heaviside(-z, 0)
			bx += bx0_p[n] * exp_cap(1j * m_p[n] * z) * np.heaviside( z, 1)
		else:
			bx += bx0_m[n] * exp_cap(1j * m_m[n] * z) * np.heaviside(-z, 0)
			bx += bx0_p[n] * exp_cap(1j * m_p[n] * z) * np.heaviside( z, 1)
	return bx

def b_perp_piecewise(z):
	for n in range(4):
		if n == 0:
			b_perp  = b_perp0_m[n] * exp_cap(1j * m_m[n] * z) * np.heaviside(-z, 0)
			b_perp += b_perp0_p[n] * exp_cap(1j * m_p[n] * z) * np.heaviside( z, 1)
		else:
			b_perp += b_perp0_m[n] * exp_cap(1j * m_m[n] * z) * np.heaviside(-z, 0)
			b_perp += b_perp0_p[n] * exp_cap(1j * m_p[n] * z) * np.heaviside( z, 1)
	return b_perp

def b_par_piecewise(z):
	for n in range(4):
		if n == 0:
			b_par  = b_par0_m[n] * exp_cap(1j * m_m[n] * z) * np.heaviside(-z, 0)
			b_par += b_par0_p[n] * exp_cap(1j * m_p[n] * z) * np.heaviside( z, 1)
		else:
			b_par += b_par0_m[n] * exp_cap(1j * m_m[n] * z) * np.heaviside(-z, 0)
			b_par += b_par0_p[n] * exp_cap(1j * m_p[n] * z) * np.heaviside( z, 1)
	return b_par

Lz = 1
nz = 8193
z_min = 0
z_max = Lz
z = np.linspace(z_min, z_max, nz)

z_min_prime = -2 / 1000
z_max_prime = 2 / 1000
z1_prime = np.linspace(0, z_max_prime, nz // 2 + 1)
z2_prime = np.linspace(z_min_prime, z_max_prime, nz)

vA_m = 0.01
vA_p = 1.0
vA0 = vA_p

alpha = 0.25 * np.pi
omega = np.pi * np.cos(alpha) * vA_p / Lz
kx = 1000 * omega / vA_p
ky = 0.5 * omega / vA_p
u0 = 1

kz_m = omega / vA_m / np.cos(alpha)
kz_p = omega / vA_p / np.cos(alpha)

m_m = np.array([
				 kz_m - ky * np.tan(alpha), \
				-kz_m - ky * np.tan(alpha), \
				 1j * np.sqrt(kx ** 2 + ky ** 2 - omega ** 2 / vA_m ** 2 + 0j), \
				-1j * np.sqrt(kx ** 2 + ky ** 2 - omega ** 2 / vA_m ** 2 + 0j), \
				])
m_p = np.array([
				 kz_p - ky * np.tan(alpha), \
				-kz_p - ky * np.tan(alpha), \
				 1j * np.sqrt(kx ** 2 + ky ** 2 - omega ** 2 / vA_p ** 2 + 0j), \
				-1j * np.sqrt(kx ** 2 + ky ** 2 - omega ** 2 / vA_p ** 2 + 0j), \
				])

nabla_par_m  = 1j * (ky * np.sin(alpha) + m_m * np.cos(alpha))
nabla_par_p  = 1j * (ky * np.sin(alpha) + m_p * np.cos(alpha))
nabla_perp_m = 1j * (ky * np.cos(alpha) - m_m * np.sin(alpha))
nabla_perp_p = 1j * (ky * np.cos(alpha) - m_p * np.sin(alpha))

L_m = nabla_par_m ** 2 + omega ** 2 / vA_m ** 2
L_p = nabla_par_p ** 2 + omega ** 2 / vA_p ** 2

ux_hat_m = -1j * kx * nabla_perp_m / (L_m - kx ** 2)
ux_hat_p = -1j * kx * nabla_perp_p / (L_p - kx ** 2)

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

# Coefficent matrix
aa = np.array([
				[         ux_hat_m[0],          ux_hat_m[3],          -ux_hat_p[1],          -ux_hat_p[2]], \
				[m_m[0] * ux_hat_m[0], m_m[3] * ux_hat_m[3], -m_p[1] * ux_hat_p[1], -m_p[2] * ux_hat_p[2]], \
				[                   1,                    1,                    -1,                    -1], \
				[              m_m[0],               m_m[3],               -m_p[1],               -m_p[2]]
				])
bb = u0 * np.array([
					         ux_hat_p[0], \
					m_p[0] * ux_hat_p[0], \
					                   1, \
					              m_p[0], \
					])
xx = np.linalg.solve(aa, bb)

u_perp0_m = np.zeros(4, dtype=np.complex)
u_perp0_p = np.zeros(4, dtype=np.complex)
u_perp0_m[0] = xx[0]
u_perp0_m[3] = xx[1]
u_perp0_p[0] = u0
u_perp0_p[1] = xx[2]
u_perp0_p[2] = xx[3]

u_perp0 = np.zeros(3, dtype=np.complex)
u_perp0[0] =  u0
u_perp0[1] = -u0 * (ux_hat[0] - ux_hat[2]) / (ux_hat[1] - ux_hat[2])
u_perp0[2] =  u0 * (ux_hat[0] - ux_hat[1]) / (ux_hat[1] - ux_hat[2])

ux0_m = ux_hat_m * u_perp0_m
ux0_p = ux_hat_p * u_perp0_p

bx0_m = nabla_perp_m * ux0_m / (1j * omega)
bx0_p = nabla_perp_p * ux0_p / (1j * omega)

b_perp0_m = nabla_perp_m * u_perp0_m / (1j * omega)
b_perp0_p = nabla_perp_p * u_perp0_p / (1j * omega)

b_par0_m = -(1j * kx * ux0_m + nabla_perp_m * u_perp0_m) / (1j * omega)
b_par0_p = -(1j * kx * ux0_p + nabla_perp_p * u_perp0_p) / (1j * omega)

ux0 = ux_hat * u_perp0
bx0 = nabla_perp0 * ux0 / (1j * omega)
b_perp0 = nabla_perp0 * u_perp0 / (1j * omega)
b_par0 = -(1j * kx * ux0 + nabla_perp0 * u_perp0) / (1j * omega)

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = 1.75 * fig_size[1]
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.4)

ax = fig.add_subplot(321)
ax.plot(z, ux_uniform(z).real, label = r'Uniform $v_A$')
ax.plot(z, ux_piecewise(z).real, '--', label = r'Piecewise constant $v_A$')
ax.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))
ax.text(0.75, 1.125, \
	r'Re$\left[u_x(0,0,z,0)\right]\, /\, u_0$', \
	fontsize = 12, \
	transform=ax.transAxes)
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,1.6))

ax = fig.add_subplot(322)
ax.plot(z1_prime, ux_uniform(z1_prime).real)
ax.plot(z2_prime, ux_piecewise(z2_prime).real, '--')
ax.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))
ax.ticklabel_format(axis = "x", style = "sci", scilimits=(0,0))
ax.text(-0.05, 1.3, \
	r"$\alpha = $" + "{:02.3f}".format(alpha / np.pi) + r"$\pi$" + '\n' + \
	r"$k_x$ = " + "{:02.1f}".format(kx / omega) + r"$\omega / v_{A+}$" + '\n' + \
	r"$k_y$ = " + "{:02.1f}".format(ky / omega) + r"$\omega / v_{A+}$",
	transform=ax.transAxes)
ax.text(0.55, 1.3, \
	r"$\omega = \pi v_{A+} \cos\alpha / L_z$" + '\n' + \
	r"$v_{A-} = $" + "{:1.2f}".format(vA_m / vA_p) + r'$\,v_{A+}$' + '\n' + \
	r"$v_{A0} = v_{A+}$", \
	transform=ax.transAxes)

ax = fig.add_subplot(323)
ax.plot(z, u_perp_uniform(z).real)
ax.plot(z, u_perp_piecewise(z).real, '--')
ax.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))
ax.text(0.75, 1.125, \
	r'Re$\left[u_\perp(0,0,z,0)\right]\, /\, u_0$', \
	fontsize = 12, \
	transform=ax.transAxes)

ax = fig.add_subplot(324)
ax.plot(z1_prime, u_perp_uniform(z1_prime).real)
ax.plot(z2_prime, u_perp_piecewise(z2_prime).real, '--')
ax.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))
ax.ticklabel_format(axis = "x", style = "sci", scilimits=(0,0))

ax = fig.add_subplot(325)
ax.plot(z, b_par_uniform(z).real)
ax.plot(z, b_par_piecewise(z).real, '--')
ax.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))
ax.text(0.75, 1.125, \
	r'Re$\left[\hat{b}_{||}(0,0,z,0)\right]\, /\, (u_0\,/\,v_{A0})$', \
	fontsize = 12, \
	transform=ax.transAxes)
ax.set_xlabel(r'$z\,/\,L_z$')

ax = fig.add_subplot(326)
ax.plot(z1_prime, b_par_uniform(z1_prime).real)
ax.plot(z2_prime, b_par_piecewise(z2_prime).real, '--')
ax.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))
ax.ticklabel_format(axis = "x", style = "sci", scilimits=(0,0))
ax.set_xlabel(r'$z\,/\,L_z$')

fig.savefig('temp_figures/real_part_large_kx.pdf', bbox_inches = 'tight')

#########################################################################################################

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = 1.75 * fig_size[1]
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)

ax = fig.add_subplot(321)
ax.plot(z, ux_uniform(z).imag, label = r'Uniform $v_A$')
ax.plot(z, ux_piecewise(z).imag, '--' , label = r'Piecewise constant $v_A$')
ax.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))
ax.text(0.75, 1.125, \
	r'Im$\left[u_x(0,0,z,0)\right]\, /\, u_0$', \
	fontsize = 12, \
	transform=ax.transAxes)
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,1.6))

ax = fig.add_subplot(322)
ax.plot(z1_prime, ux_uniform(z1_prime).imag)
ax.plot(z2_prime, ux_piecewise(z2_prime).imag, '--')
ax.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))
ax.ticklabel_format(axis = "x", style = "sci", scilimits=(0,0))
ax.text(-0.05, 1.3, \
	r"$\alpha = $" + "{:02.3f}".format(alpha / np.pi) + r"$\pi$" + '\n' + \
	r"$k_x$ = " + "{:02.1f}".format(kx / omega) + r"$\omega / v_{A+}$" + '\n' + \
	r"$k_y$ = " + "{:02.1f}".format(ky / omega) + r"$\omega / v_{A+}$",
	transform=ax.transAxes)
ax.text(0.55, 1.3, \
	r"$\omega = \pi v_{A+} \cos\alpha / L_z$" + '\n' + \
	r"$v_{A-} = $" + "{:1.2f}".format(vA_m / vA_p) + r'$\,v_{A+}$' + '\n' + \
	r"$v_{A0} = v_{A+}$", \
	transform=ax.transAxes)

ax = fig.add_subplot(323)
ax.plot(z, u_perp_uniform(z).imag)
ax.plot(z, u_perp_piecewise(z).imag, '--')
ax.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))
ax.text(0.75, 1.125, \
	r'Im$\left[u_\perp(0,0,z,0)\right]\, /\, u_0$', \
	fontsize = 12, \
	transform=ax.transAxes)

ax = fig.add_subplot(324)
ax.plot(z1_prime, u_perp_uniform(z1_prime).imag)
ax.plot(z2_prime, u_perp_piecewise(z2_prime).imag, '--')
ax.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))
ax.ticklabel_format(axis = "x", style = "sci", scilimits=(0,0))

ax = fig.add_subplot(325)
ax.plot(z, b_par_uniform(z).imag)
ax.plot(z, b_par_piecewise(z).imag, '--')
ax.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))
ax.text(0.75, 1.125, \
	r'Im$\left[\hat{b}_{||}(0,0,z,0)\right]\, /\, (u_0\,/\,v_{A+})$', \
	fontsize = 12, \
	transform=ax.transAxes)
ax.set_xlabel(r'$z\,/\,L_z$')

ax = fig.add_subplot(326)
ax.plot(z1_prime, b_par_uniform(z1_prime).imag)
ax.plot(z2_prime, b_par_piecewise(z2_prime).imag, '--')
ax.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))
ax.ticklabel_format(axis = "x", style = "sci", scilimits=(0,0))
ax.set_xlabel(r'$z\,/\,L_z$')

fig.savefig('temp_figures/imag_part_large_kx.pdf', bbox_inches = 'tight')

plt.show()