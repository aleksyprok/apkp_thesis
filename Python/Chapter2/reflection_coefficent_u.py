import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel2
from scipy.integrate import solve_ivp

def vA(z):
	z0 = np.array(z)
	va = np.zeros_like(z0)
	va += vA0 * np.exp(z0 / (2 * h)) * (z0 <  0)
	va += vA0                        * (z0 >= 0)
	return va

def xi(z):
	return 2 * h * omega / vA(z)

def u_ana(z):
	z0 = np.array(z)
	u = np.zeros_like(z0, dtype = complex)
	u += c2 * hankel2(0, xi(z0))                                  *  (z0 <  0)
	u += u0 * (np.exp(1j * kz * z0) + c4 * np.exp(-1j * kz * z0)) *  (z0 >= 0)
	return u

def du_dz_ana(z):
	z0 = np.array(z)
	u = np.zeros_like(z0, dtype = complex)
	u += c2 * (omega / vA(z0)) * hankel2(1, xi(z0))                         *  (z0 <  0)
	u += 1j * kz * u0 * (np.exp(1j * kz * z0) - c4 * np.exp(-1j * kz * z0)) *  (z0 >= 0)
	return u

def wave_eqn(z, U):
	# U[0] = u
	# U[1] = u'
	return np.array([U[1], -(omega / vA(z)) ** 2 * U[0]])

def R_approx(xi0):
	return 1 - np.pi * xi0

h = 150 * 1e3
omega = 2 * np.pi * 1e-1
vA0 = 1 * 1e6
xi0 = 2 * h * omega / vA0
kz = omega / vA0

u0 = 1
c2 = 2 * u0 / (hankel2(0,xi0) - 1j * hankel2(1, xi0))
c4 = u0 * (hankel2(0,xi0) + 1j * hankel2(1, xi0)) / (hankel2(0,xi0) - 1j * hankel2(1, xi0))

z_min = -2 * 1e6
z_max =  5 * 1e6
nz = 8192
z = np.linspace(z_min, z_max, nz)

U0 = np.array([u_ana(z_min), du_dz_ana(z_min)])
sol = solve_ivp(wave_eqn, [z_min, z_max], U0, t_eval = z, rtol = 1e-5, atol = 1e-10)

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = fig_size[1] / 2
fig.set_size_inches(fig_size)

ax = fig.add_subplot(121)
ax.plot(z * 1e-6, np.real(u_ana(z)), label = 'Analytic')
ax.plot(z * 1e-6, np.real(sol.y[0,:]), ':', label = 'Numerical')
ax.set_xlabel(r'$z\ $ (Mm)')
ax.set_title(r'Re$(u)$ / $u_0$')
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1,1.3))
ax.text(0.25, 0.35, \
	'$f$ = ' + '{:1.1f}'.format(omega / (2 * np.pi)) + ' Hz' + '\n' + \
	r'$v_{A0}$ = ' + '{:1.0f}'.format(vA0 / 1e6) + r' Mm$\,s^{-1}$', \
	transform=ax.transAxes)

ax = fig.add_subplot(122)
ax.plot(z * 1e-6, np.imag(u_ana(z)))
ax.plot(z * 1e-6, np.imag(sol.y[0,:]), ':')
ax.set_xlabel(r'$z\ $ (Mm)')
ax.set_title(r'Im$(u)$ / $u_0$')
ax.text(0.4, 0.4, \
	'$h$ = ' + '{:1.0f}'.format(h / (1e3)) + ' km', \
	transform=ax.transAxes)

fig.savefig('temp_figures/reflection_coefficent_u.pdf', bbox_inches = 'tight')

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = fig_size[1] / 2
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.2)

ax = fig.add_subplot(121)
ax.plot(z * 1e-6, vA(z) * 1e-6)
ax.set_xlabel(r'$z\ $ (Mm)')
ax.set_title(r'$v_A\ $ (Mm$\,s^{-1}$) ')
ax.text(0.425, 0.35, \
	'$h$ = ' + '{:1.0f}'.format(h / (1e3)) + ' km' + '\n' + \
	r'$v_{A0}$ = ' + '{:1.0f}'.format(vA0 / 1e6) + r' Mm$\,s^{-1}$', \
	transform=ax.transAxes)

f_min = -4
f_max =  1
nf = 8192
f = np.logspace(f_min, f_max, nf)
omega = 2 * np.pi * f
xi0 = 2 * h * omega / vA0
c4 = u0 * (hankel2(0,xi0) + 1j * hankel2(1, xi0)) / (hankel2(0,xi0) - 1j * hankel2(1, xi0))
ax = fig.add_subplot(122)
ax.loglog(f, 1 - np.abs(c4 / u0), label = 'Full solution')
ax.loglog(f, 1 - R_approx(xi0), ':', label = 'Approximation')
ylim = ax.get_ylim()
ax.set_ylim(ylim[0], 2)
ax.set_xlabel(r'$f\ $ (Hz)')
ax.set_title(r'$1 - R$ ')
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels)

fig.savefig('temp_figures/reflection_coefficent.pdf', bbox_inches = 'tight')

plt.show(block = False)