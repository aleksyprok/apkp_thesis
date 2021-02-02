import numpy as np
import matplotlib.pyplot as plt

def m(t):
	return np.floor(t * vA0 / Lz)

def f_driv(t):
	t = np.array(t)
	return u0 * np.sin(np.pi * t / P)

def u_ana(z,t):
	u = np.zeros_like(t, dtype = float)
	for k in np.arange(m(t_max)+1):
		theta_k = t - (-1) ** k * z / vA0 - (2 * k + 1) * lz / vA0
		u += u0 * (-1) ** k * R ** k * np.heaviside(theta_k, 1) * f_driv(theta_k)
	return u

def b_ana(z,t):
	b = np.zeros_like(t, dtype = float)
	for k in np.arange(m(t_max)+1):
		theta_k = t - (-1) ** k * z / vA0 - (2 * k + 1) * lz / vA0
		b -= B0 / vA0 * u0 * R ** k * np.heaviside(theta_k, 1) * f_driv(theta_k)
	return b

def zp_ana(z,t):
	return u_ana(z,t) + vA0 * b_ana(z,t) / B0

def zm_ana(z,t):
	return u_ana(z,t) - vA0 * b_ana(z,t) / B0

P = 1
vA0 = 1
u0 = 1
R = 0.75
B0 = 1

t_min = 0
t_max = 15
nt = 1024
t = np.linspace(t_min, t_max, nt)

lz = 0.5
Lz = 2 * lz
z_min = -lz
z_max = lz
nz = 1024
z = np.linspace(z_min, z_max, nz)

T, Z = np.meshgrid(t,z)

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = fig_size[1] * 1.75
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.3)

ax = fig.add_subplot(321)
ax.plot(t, u_ana(-lz, t), label = 'Full solution')
ax.plot(t, f_driv(t) + (1 - R) * zp_ana(-lz, t) / 2, '--', \
	label = r'$f_{driv}(t) + \frac{1}{2}(1 - R)\mathcal{Z}^+(-l_z,t)$')
ax.plot([], [], 'r:', label = r'$\frac{1}{2}(1 - R)\mathcal{Z}^-(l_z,t)$')
ax.set_title(r'$u(-l_z,t) / u_0$')
ax.text(1.05, 1.05, \
	'$R$ = ' + '{:1.2f}'.format(R), \
	transform=ax.transAxes)
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.25,1.65))

ax = fig.add_subplot(322)
ax.plot(t, u_ana(lz, t))
ax.plot([], [])
ax.plot(t, (1 - R) * zm_ana(lz, t) / 2, 'r:')
ax.set_title(r'$u(l_z,t) / u_0$')

ax = fig.add_subplot(323)
ax.plot(t, u_ana(0, t))
ax.set_title(r'$u(0,t) / u_0$')

ax = fig.add_subplot(324)
cp = ax.contourf(T, Z, u_ana(Z,T), levels=100)
plt.colorbar(cp)
ax.set_title(r'$u(z,t) / u_0$')
ax.set_ylabel(r'$z / L_z$')

ax = fig.add_subplot(325)
ax.plot(t, b_ana(0, t))
ax.set_title(r'$b(0,t) / b_0$')
ax.set_xlabel(r'$t / t_0$')

ax = fig.add_subplot(326)
cp = ax.contourf(T, Z, b_ana(Z,T), levels=100)
plt.colorbar(cp)
ax.set_title(r'$b(z,t) / b_0$')
ax.set_xlabel(r'$t / t_0$')
ax.set_ylabel(r'$z / L_z$')

fig.savefig('temp_figures/leaky_wave.png', bbox_inches = 'tight', dpi=150)


plt.show(block = False)