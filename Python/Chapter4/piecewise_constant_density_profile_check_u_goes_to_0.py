import numpy as np
import matplotlib.pyplot as plt

n_iters = 100
vA_m_array = np.logspace(-4, -1, n_iters)
u_perp_m1_array = np.zeros(n_iters, dtype=np.complex)
u_perp_m4_array = np.zeros(n_iters, dtype=np.complex)
ux_m4_array = np.zeros(n_iters, dtype=np.complex)
ux_m1_array = np.zeros(n_iters, dtype=np.complex)
ux_m4_array = np.zeros(n_iters, dtype=np.complex)

for iters in range(n_iters):

	Lz = 1
	nz = 1024
	z_min = -Lz
	z_max = Lz
	dz = (z_max - z_min) / (nz - 1)
	z = np.linspace(z_min, z_max, nz)

	vA_m = vA_m_array[iters]
	vA_p = 1.0

	alpha = 0.25 * np.pi
	omega = np.pi * np.cos(alpha) * vA_p / Lz
	kx = 10 * omega / vA_p
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

	u_perp0_m = np.zeros(4, dtype=np.complex)
	u_perp0_p = np.zeros(4, dtype=np.complex)
	u_perp0_m[0] = xx[0]
	u_perp0_m[3] = xx[1]
	u_perp0_p[0] = u0
	u_perp0_p[1] = xx[2]
	u_perp0_p[2] = xx[3]

	ux0_m = ux_hat_m * u_perp0_m
	ux0_p = ux_hat_p * u_perp0_p

	u_perp_m1_array[iters] = u_perp0_m[0]
	u_perp_m4_array[iters] = u_perp0_m[3]
	ux_m1_array[iters] = ux0_m[0]
	ux_m4_array[iters] = ux0_m[3]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(vA_m_array, np.abs(u_perp_m1_array))
ax.set_xlabel(r'$v_{A-}$')
ax.set_title(r'$u_{\perp 1-}$')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(vA_m_array, np.abs(u_perp_m4_array))
ax.set_xlabel(r'$v_{A-}$')
ax.set_title(r'$u_{\perp 4-}$')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(vA_m_array, np.abs(ux_m1_array))
ax.set_xlabel(r'$v_{A-}$')
ax.set_title(r'$u_{x 1-}$')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(vA_m_array, np.abs(ux_m4_array))
ax.set_xlabel(r'$v_{A-}$')
ax.set_title(r'$u_{x 4-}$')

plt.show()