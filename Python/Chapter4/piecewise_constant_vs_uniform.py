import numpy as np
import matplotlib.pyplot as plt

def ux_piecewise(z):
	for n in range(3):
		z_exp = z * np.heaviside( z, 1)
		if n == 0:
			ux = ux0_p[n] * np.exp(1j * m_p[n] * z_exp) * np.heaviside( z, 1)
		else:
			ux += ux0_p[n] * np.exp(1j * m_p[n] * z_exp) * np.heaviside( z, 1)
	for n in [0, 3]:
		z_exp = z * np.heaviside(-z, 0)
		if n == 0:
			ux += ux0_m[n] * np.exp(1j * m_m[n] * z_exp) * np.heaviside(-z, 0)
		else:
			ux += ux0_m[n] * np.exp(1j * m_m[n] * z_exp) * np.heaviside(-z, 0)
	return ux

def u_perp_piecewise(z):
	for n in range(3):
		z_exp = z * np.heaviside( z, 1)
		if n == 0:
			u_perp = u_perp0_p[n] * np.exp(1j * m_p[n] * z_exp) * np.heaviside( z, 1)
		else:
			u_perp += u_perp0_p[n] * np.exp(1j * m_p[n] * z_exp) * np.heaviside( z, 1)
	for n in [0, 3]:
		z_exp = z * np.heaviside(-z, 0)
		if n == 0:
			u_perp += u_perp0_m[n] * np.exp(1j * m_m[n] * z_exp) * np.heaviside(-z, 0)
		else:
			u_perp += u_perp0_m[n] * np.exp(1j * m_m[n] * z_exp) * np.heaviside(-z, 0)
	return u_perp

def bx_piecewise(z):
	for n in range(3):
		z_exp = z * np.heaviside( z, 1)
		if n == 0:
			bx = bx0_p[n] * np.exp(1j * m_p[n] * z_exp) * np.heaviside( z, 1)
		else:
			bx += bx0_p[n] * np.exp(1j * m_p[n] * z_exp) * np.heaviside( z, 1)
	for n in [0, 3]:
		z_exp = z * np.heaviside(-z, 0)
		if n == 0:
			bx += bx0_m[n] * np.exp(1j * m_m[n] * z_exp) * np.heaviside(-z, 0)
		else:
			bx += bx0_m[n] * np.exp(1j * m_m[n] * z_exp) * np.heaviside(-z, 0)
	return bx

def b_perp_piecewise(z):
	for n in range(3):
		z_exp = z * np.heaviside( z, 1)
		if n == 0:
			b_perp = b_perp0_p[n] * np.exp(1j * m_p[n] * z_exp) * np.heaviside( z, 1)
		else:
			b_perp += b_perp0_p[n] * np.exp(1j * m_p[n] * z_exp) * np.heaviside( z, 1)
	for n in [0, 3]:
		z_exp = z * np.heaviside(-z, 0)
		if n == 0:
			b_perp += b_perp0_m[n] * np.exp(1j * m_m[n] * z_exp) * np.heaviside(-z, 0)
		else:
			b_perp += b_perp0_m[n] * np.exp(1j * m_m[n] * z_exp) * np.heaviside(-z, 0)
	return b_perp

def b_par_piecewise(z):
	for n in range(3):
		z_exp = z * np.heaviside( z, 1)
		if n == 0:
			b_par = b_par0_p[n] * np.exp(1j * m_p[n] * z_exp) * np.heaviside( z, 1)
		else:
			b_par += b_par0_p[n] * np.exp(1j * m_p[n] * z_exp) * np.heaviside( z, 1)
	for n in [0, 3]:
		z_exp = z * np.heaviside(-z, 0)
		if n == 0:
			b_par += b_par0_m[n] * np.exp(1j * m_m[n] * z_exp) * np.heaviside(-z, 0)
		else:
			b_par += b_par0_m[n] * np.exp(1j * m_m[n] * z_exp) * np.heaviside(-z, 0)
	return b_par


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

Lz = 1
nz = 1025
z_min = 0
z_max = Lz
# dz = (z_max - z_min) / (nz - 1)
# z1 = np.linspace(z_min, z_max, nz)
# z2 = np.linspace(0, z_max, nz // 2 + 1)
z = np.linspace(z_min, z_max, nz)

z_min_prime = -1 / 1000
z_max_prime = 1 / 1000
z1_prime = np.linspace(z_min_prime, z_max_prime, nz)
z2_prime = np.linspace(0, z_max_prime, nz // 2 + 1)

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
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)

ax = fig.add_subplot(321)
ax.plot(z, ux_piecewise(z).real)
ax.plot(z, ux_uniform(z).real)
ax.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))
yrange = ax.get_ylim()
ax.plot([0,0],yrange, 'k:')
ax.set_ylim(yrange)
ax.set_title(r'Re$[u_x(0,0,z,0)] / u_0$')

ax = fig.add_subplot(322)
ax.plot(z, u_perp_piecewise(z).real)
ax.plot(z, u_perp_uniform(z).real)
ax.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))
yrange = ax.get_ylim()
ax.plot([0,0],yrange, 'k:')
ax.set_ylim(yrange)
ax.set_title(r'Re$[u_\perp(0,0,z,0)] / u_0$')

ax = fig.add_subplot(323)
ax.plot(z, bx_piecewise(z).real)
ax.plot(z, bx_uniform(z).real)
ax.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))
yrange = ax.get_ylim()
ax.plot([0,0],yrange, 'k:')
ax.set_ylim(yrange)
ax.set_title(r'Re$[b_x(0,0,z,0)] / u_0$')

ax = fig.add_subplot(324)
ax.plot(z, b_perp_piecewise(z).real)
ax.plot(z, b_perp_uniform(z).real)
ax.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))
yrange = ax.get_ylim()
ax.plot([0,0],yrange, 'k:')
ax.set_ylim(yrange)
ax.set_title(r'Re$[b_\perp(0,0,z,0)] / u_0$')

ax = fig.add_subplot(325)
ax.plot(z, b_par_piecewise(z).real)
ax.plot(z, b_par_uniform(z).real)
ax.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))
yrange = ax.get_ylim()
ax.plot([0,0],yrange, 'k:')
ax.set_ylim(yrange)
ax.set_title(r'Re$[b_{||}(0,0,z,0)] / u_0$')

ax = fig.add_subplot(326)
ax.plot(z1_prime, b_par_piecewise(z1_prime).real)
ax.plot(z2_prime, b_par_uniform(z2_prime).real)
ax.ticklabel_format(axis = "x", style = "sci", scilimits=(0,0))
ax.ticklabel_format(axis = "y", style = "sci", scilimits=(0,0))
yrange = ax.get_ylim()
ax.plot([0,0],yrange, 'k:')
ax.set_ylim(yrange)
ax.set_title(r'Re$[b_{||}(0,0,z,0)] / u_0$')

###########################################################################

# fig = plt.figure()
# fig_size = fig.get_size_inches()
# fig_size[1] = 1.75 * fig_size[1]
# fig.set_size_inches(fig_size)
# plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)

# ax = fig.add_subplot(321)
# ax.plot(z, ux_piecewise(z).imag)
# ax.plot(z, ux_uniform(z).imag)
# yrange = ax.get_ylim()
# ax.plot([0,0],yrange, 'k:')
# ax.set_ylim(yrange)
# ax.set_title(r'Re$[u_x(0,0,z,0)] / u_0$')

# ax = fig.add_subplot(322)
# ax.plot(z, u_perp_piecewise(z).imag)
# ax.plot(z, u_perp_uniform(z).imag)
# yrange = ax.get_ylim()
# ax.plot([0,0],yrange, 'k:')
# ax.set_ylim(yrange)
# ax.set_title(r'Re$[u_\perp(0,0,z,0)] / u_0$')

# ax = fig.add_subplot(323)
# ax.plot(z, bx_piecewise(z).imag)
# ax.plot(z, bx_uniform(z).imag)
# yrange = ax.get_ylim()
# ax.plot([0,0],yrange, 'k:')
# ax.set_ylim(yrange)
# ax.set_title(r'Re$[b_x(0,0,z,0)] / u_0$')

# ax = fig.add_subplot(324)
# ax.plot(z, b_perp_piecewise(z).imag)
# ax.plot(z, b_perp_uniform(z).imag)
# yrange = ax.get_ylim()
# ax.plot([0,0],yrange, 'k:')
# ax.set_ylim(yrange)
# ax.set_title(r'Re$[b_\perp(0,0,z,0)] / u_0$')

# ax = fig.add_subplot(325)
# ax.plot(z, b_par_piecewise(z).imag)
# ax.plot(z, b_par_uniform(z).imag)
# yrange = ax.get_ylim()
# ax.plot([0,0],yrange, 'k:')
# ax.set_ylim(yrange)
# ax.set_title(r'Re$[b_{||}(0,0,z,0)] / u_0$')

# ax = fig.add_subplot(326)
# ax.plot(z1_prime, b_par_piecewise(z1_prime).imag)
# ax.plot(z2_prime, b_par_uniform(z2_prime).imag)
# ax.ticklabel_format(axis = "x", style = "sci", scilimits=(0,0))
# yrange = ax.get_ylim()
# ax.plot([0,0],yrange, 'k:')
# ax.set_ylim(yrange)
# ax.set_title(r'Re$[b_{||}(0,0,z,0)] / u_0$')

print(r'$k_x^2 =$', kx ** 2)
print(r'$\omega^2 \tan^2(\alpha) / (v_{A+} v_{A-}) =$', omega ** 2 * np.tan(alpha) ** 2 / vA_p / vA_m)

plt.show()