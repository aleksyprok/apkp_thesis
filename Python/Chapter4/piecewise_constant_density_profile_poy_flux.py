import numpy as np
import matplotlib.pyplot as plt

def ux(z):
	for n in range(4):
		if n == 0:
			ux  = ux_m[n] * np.exp(1j * m_m[n] * z) * (z <  0)
			ux += ux_p[n] * np.exp(1j * m_p[n] * z) * (z >= 0)
		else:
			ux += ux_m[n] * np.exp(1j * m_m[n] * z) * (z <  0)
			ux += ux_p[n] * np.exp(1j * m_p[n] * z) * (z >= 0)
	return ux

def u_perp(z):
	for n in range(4):
		if n == 0:
			u_perp  = u_perp_m[n] * np.exp(1j * m_m[n] * z) * (z <  0)
			u_perp += u_perp_p[n] * np.exp(1j * m_p[n] * z) * (z >= 0)
		else:
			u_perp += u_perp_m[n] * np.exp(1j * m_m[n] * z) * (z <  0)
			u_perp += u_perp_p[n] * np.exp(1j * m_p[n] * z) * (z >= 0)
	return u_perp

def bx(z):
	for n in range(4):
		if n == 0:
			bx  = bx_m[n] * np.exp(1j * m_m[n] * z) * (z <  0)
			bx += bx_p[n] * np.exp(1j * m_p[n] * z) * (z >= 0)
		else:
			bx += bx_m[n] * np.exp(1j * m_m[n] * z) * (z <  0)
			bx += bx_p[n] * np.exp(1j * m_p[n] * z) * (z >= 0)
	return bx

def b_perp(z):
	for n in range(4):
		if n == 0:
			b_perp  = b_perp_m[n] * np.exp(1j * m_m[n] * z) * (z <  0)
			b_perp += b_perp_p[n] * np.exp(1j * m_p[n] * z) * (z >= 0)
		else:
			b_perp += b_perp_m[n] * np.exp(1j * m_m[n] * z) * (z <  0)
			b_perp += b_perp_p[n] * np.exp(1j * m_p[n] * z) * (z >= 0)
	return b_perp

def b_par(z):
	for n in range(4):
		if n == 0:
			b_par  = b_par_m[n] * np.exp(1j * m_m[n] * z) * (z <  0)
			b_par += b_par_p[n] * np.exp(1j * m_p[n] * z) * (z >= 0)
		else:
			b_par += b_par_m[n] * np.exp(1j * m_m[n] * z) * (z <  0)
			b_par += b_par_p[n] * np.exp(1j * m_p[n] * z) * (z >= 0)
	return b_par

def dux(z):
	for n in range(4):
		if n == 0:
			dux_ana_dz  = 1j * m_m[n] * ux_m[n] * np.exp(1j * m_m[n] * z) * np.heaviside(-z, 1)
			dux_ana_dz += 1j * m_p[n] * ux_p[n] * np.exp(1j * m_p[n] * z) * np.heaviside( z, 0)
		else:
			dux_ana_dz += 1j * m_m[n] * ux_m[n] * np.exp(1j * m_m[n] * z) * np.heaviside(-z, 1)
			dux_ana_dz += 1j * m_p[n] * ux_p[n] * np.exp(1j * m_p[n] * z) * np.heaviside( z, 0)
	return dux_ana_dz

def du_perp(z):
	for n in range(4):
		if n == 0:
			du_perp  = 1j * m_m[n] * u_perp_m[n] * np.exp(1j * m_m[n] * z) * np.heaviside(-z, 1)
			du_perp += 1j * m_p[n] * u_perp_p[n] * np.exp(1j * m_p[n] * z) * np.heaviside( z, 0)
		else:
			du_perp += 1j * m_m[n] * u_perp_m[n] * np.exp(1j * m_m[n] * z) * np.heaviside(-z, 1)
			du_perp += 1j * m_p[n] * u_perp_p[n] * np.exp(1j * m_p[n] * z) * np.heaviside( z, 0)
	return du_perp

Lz = 1
nz = 8192
z_min = -10 * Lz
z_max = 10 * Lz
dz = (z_max - z_min) / (nz - 1)
z = np.linspace(z_min, z_max, nz)

vA_m = 0.1
vA_p = 1.0

alpha = 0.25 * np.pi
omega = np.pi * np.cos(alpha) * vA_p / Lz
kx = 1 * omega / vA_p
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

u_perp_m = np.zeros(4, dtype=np.complex)
u_perp_p = np.zeros(4, dtype=np.complex)
u_perp_m[0] = xx[0]
u_perp_m[3] = xx[1]
u_perp_p[0] = u0
u_perp_p[1] = xx[2]
u_perp_p[2] = xx[3]

ux_m = ux_hat_m * u_perp_m
ux_p = ux_hat_p * u_perp_p

bx_m = nabla_par_m / (1j * omega) * ux_m
bx_p = nabla_par_p / (1j * omega) * ux_p

b_perp_m = nabla_par_m / (1j * omega) * u_perp_m
b_perp_p = nabla_par_p / (1j * omega) * u_perp_p

b_par_m = -1 / (1j * omega) * (1j * kx * ux_m + nabla_perp_m * u_perp_m)
b_par_p = -1 / (1j * omega) * (1j * kx * ux_p + nabla_perp_p * u_perp_p)

poy_flux = -0.5 * np.real(np.conj(u_perp(z)) * b_par(z) * np.sin(alpha) + (\
						  np.conj(ux(z))     * bx(z) + \
						  np.conj(u_perp(z)) * b_perp(z)) * np.cos(alpha))

print('vAp^2(kx^2+ky^2) =', vA_p ** 2 * (kx ** 2 + ky ** 2))
print('vAm^2(kx^2+ky^2) =', vA_m ** 2 * (kx ** 2 + ky ** 2))
print('omega^2 =', omega ** 2)
print('kx  =', kx)
print('ky  =', ky)
print('alpha  =', alpha)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, poy_flux)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(z, ux(z).real)
# ax.plot(z, ux(z).imag)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(z, u_perp(z).real)
# ax.plot(z, u_perp(z).imag)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(z, dux(z).real)
# ax.plot(z, dux(z).imag)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(z, du_perp(z).real)
# ax.plot(z, du_perp(z).imag)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(z, bx(z).real)
# ax.plot(z, bx(z).imag)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(z, b_perp(z).real)
# ax.plot(z, b_perp(z).imag)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(z, b_par(z).real)
# ax.plot(z, b_par(z).imag)


plt.show()