import numpy as np
import matplotlib.pyplot as plt

def ux_ana(z):
	for n in range(4):
		if n == 0:
			ux  = ux0_m[n] * np.exp(1j * m_m[n] * z) * np.heaviside(-z, 1)
			ux += ux0_p[n] * np.exp(1j * m_p[n] * z) * np.heaviside( z, 0)
		else:
			ux += ux0_m[n] * np.exp(1j * m_m[n] * z) * np.heaviside(-z, 1)
			ux += ux0_p[n] * np.exp(1j * m_p[n] * z) * np.heaviside( z, 0)
	return ux

def u_perp_ana(z):
	for n in range(4):
		if n == 0:
			u_perp  = u_perp0_m[n] * np.exp(1j * m_m[n] * z) * np.heaviside(-z, 1)
			u_perp += u_perp0_p[n] * np.exp(1j * m_p[n] * z) * np.heaviside( z, 0)
		else:
			u_perp += u_perp0_m[n] * np.exp(1j * m_m[n] * z) * np.heaviside(-z, 1)
			u_perp += u_perp0_p[n] * np.exp(1j * m_p[n] * z) * np.heaviside( z, 0)
	return u_perp

def dux_ana(z):
	for n in range(4):
		if n == 0:
			dux_ana_dz  = 1j * m_m[n] * ux0_m[n] * np.exp(1j * m_m[n] * z) * np.heaviside(-z, 1)
			dux_ana_dz += 1j * m_p[n] * ux0_p[n] * np.exp(1j * m_p[n] * z) * np.heaviside( z, 0)
		else:
			dux_ana_dz += 1j * m_m[n] * ux0_m[n] * np.exp(1j * m_m[n] * z) * np.heaviside(-z, 1)
			dux_ana_dz += 1j * m_p[n] * ux0_p[n] * np.exp(1j * m_p[n] * z) * np.heaviside( z, 0)
	return dux_ana_dz

def du_perp_ana(z):
	for n in range(4):
		if n == 0:
			du_perp_dz  = 1j * m_m[n] * u_perp0_m[n] * np.exp(1j * m_m[n] * z) * np.heaviside(-z, 1)
			du_perp_dz += 1j * m_p[n] * u_perp0_p[n] * np.exp(1j * m_p[n] * z) * np.heaviside( z, 0)
		else:
			du_perp_dz += 1j * m_m[n] * u_perp0_m[n] * np.exp(1j * m_m[n] * z) * np.heaviside(-z, 1)
			du_perp_dz += 1j * m_p[n] * u_perp0_p[n] * np.exp(1j * m_p[n] * z) * np.heaviside( z, 0)
	return du_perp_dz

def uxA_ana(z):
	for n in range(2):
		if n == 0:
			uxA  = ux0_m[n] * np.exp(1j * m_m[n] * z) * np.heaviside(-z, 1)
			uxA += ux0_p[n] * np.exp(1j * m_p[n] * z) * np.heaviside( z, 0)
		else:
			uxA += ux0_m[n] * np.exp(1j * m_m[n] * z) * np.heaviside(-z, 1)
			uxA += ux0_p[n] * np.exp(1j * m_p[n] * z) * np.heaviside( z, 0)
	return uxA

def uxf_ana(z):
	for n in range(2,4):
		if n == 2:
			uxf  = ux0_m[n] * np.exp(1j * m_m[n] * z) * np.heaviside(-z, 1)
			uxf += ux0_p[n] * np.exp(1j * m_p[n] * z) * np.heaviside( z, 0)
		else:
			uxf += ux0_m[n] * np.exp(1j * m_m[n] * z) * np.heaviside(-z, 1)
			uxf += ux0_p[n] * np.exp(1j * m_p[n] * z) * np.heaviside( z, 0)
	return uxf

Lz = 1
nz = 1024
z_min = -Lz
z_max = Lz
dz = (z_max - z_min) / (nz - 1)
z = np.linspace(z_min, z_max, nz)

vA_m = 0.1
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

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = 1.75 * fig_size[1]
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)

ax = fig.add_subplot(321)
ax.plot(z, ux_ana(z).real, label = 'Real part')
ax.plot(z, ux_ana(z).imag, label = 'Imag part')
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,1.45))
yrange = ax.get_ylim()
ax.plot([0,0],yrange, 'k:')
ax.set_ylim(yrange)
ax.set_title(r'$u_x(0,0,z,0) / u_0$')

ax = fig.add_subplot(322)
ax.plot(z, dux_ana(z).real)
ax.plot(z, dux_ana(z).imag)
yrange = ax.get_ylim()
ax.plot([0,0],yrange, 'k:')
ax.set_ylim(yrange)
ax.set_title(r'$[\partial u_x(0,0,z,0) / \partial z]\ /\ (u_0\, /\, L_z)$')
ax.text(-0.05, 1.15, \
	r"$\alpha = $" + "{:02.3f}".format(alpha / np.pi) + r"$\pi$" + '\n' + \
	r"$k_x$ = " + "{:02.1f}".format(kx / omega) + r"$\omega / v_{A+}$" + '\n' + \
	r"$k_y$ = " + "{:02.1f}".format(ky / omega) + r"$\omega / v_{A+}$",
	transform=ax.transAxes)
ax.text(0.55, 1.225, \
	r"$\omega = \pi v_{A+} \cos\alpha / L_z$" + '\n' + \
	r"$v_{A-} = $" + "{:1.1f}".format(vA_m / vA_p) + r'$\,v_{A+}$', \
	transform=ax.transAxes)


ax = fig.add_subplot(323)
ax.plot(z, u_perp_ana(z).real)
ax.plot(z, u_perp_ana(z).imag)
yrange = ax.get_ylim()
ax.plot([0,0],yrange, 'k:')
ax.set_ylim(yrange)
ax.set_title(r'$u_\perp(0,0,z,0) / u_0$')

ax = fig.add_subplot(324)
ax.plot(z, du_perp_ana(z).real)
ax.plot(z, du_perp_ana(z).imag)
yrange = ax.get_ylim()
ax.plot([0,0],yrange, 'k:')
ax.set_ylim(yrange)
ax.set_title(r'$[\partial u_\perp(0,0,z,0) / \partial z]\ /\ (u_0\, /\, L_z)$')

ax = fig.add_subplot(325)
ax.plot(z, uxA_ana(z).real)
ax.plot(z, uxA_ana(z).imag)
yrange = ax.get_ylim()
ax.plot([0,0],yrange, 'k:')
ax.set_ylim(yrange)
ax.set_title(r'Alfv$\grave{e}$n wave component of $u_x/u_0$')
ax.set_xlabel(r'$z / L_z$')

ax = fig.add_subplot(326)
ax.plot(z, uxf_ana(z).real)
ax.plot(z, uxf_ana(z).imag)
yrange = ax.get_ylim()
ax.plot([0,0],yrange, 'k:')
ax.set_ylim(yrange)
# ax.set_xlim(-0.1,0.1)
ax.set_title(r'Fast wave component of $u_x/u_0$')
ax.set_xlabel(r'$z / L_z$')

fig.savefig('temp_figures/piecewise_constant_density_profile.png', bbox_inches = 'tight', dpi=150)

plt.show(block = False)

print('vAp^2(kx^2+ky^2) =', vA_p ** 2 * (kx ** 2 + ky ** 2))
print('omega^2         =', omega ** 2)