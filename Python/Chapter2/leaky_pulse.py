import numpy as np
import matplotlib.pyplot as plt

def m(t):
	return np.floor(t * vA0 / Lz)

def f_driv(t):
	t = np.array(t)
	return u0 * np.sin(2 * np.pi * t / P) ** 2 * (t <= P / 2)

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
R = 0.5
B0 = 1

t_min = 0
t_max = 5
nt = 1024
t = np.linspace(t_min, t_max, nt)

lz = 0.5
Lz = 2 * lz
z_min = -lz
z_max = lz
nz = 1024
z = np.linspace(z_min, z_max, nz)

T, Z = np.meshgrid(t,z)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(t, f_driv(t))

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = fig_size[1] * 2
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

ax = fig.add_subplot(421)
ax.plot(t, u_ana(0, t))
ax.set_title(r'$u(0,t) / u_0$')
# ax.set_xlabel(r'$t / t_0$')
ax.text(1.05, 1.05, \
	'$R$ = ' + '{:1.1f}'.format(R), \
	transform=ax.transAxes)

ax = fig.add_subplot(422)
cp = ax.contourf(T, Z, u_ana(Z,T), levels=100)
plt.colorbar(cp)
ax.set_title(r'$u(z,t) / u_0$')
# ax.set_xlabel(r'$t / t_0$')
ax.set_ylabel(r'$z / L_z$')

ax = fig.add_subplot(423)
ax.plot(t, b_ana(0, t))
ax.set_title(r'$b(0,t) / b_0$')
# ax.set_xlabel(r'$t / t_0$')

ax = fig.add_subplot(424)
cp = ax.contourf(T, Z, b_ana(Z,T), levels=100)
plt.colorbar(cp)
ax.set_title(r'$b(z,t) / b_0$')
# ax.set_xlabel(r'$t / t_0$')
ax.set_ylabel(r'$z / L_z$')

ax = fig.add_subplot(425)
ax.plot(t, zp_ana(0, t))
ax.set_title(r'$\mathcal{Z}^+(0,t) / u_0$')
# ax.set_xlabel(r'$t / t_0$')

ax = fig.add_subplot(426)
cp = ax.contourf(T, Z, zp_ana(Z,T), levels=100)
ax.set_title(r'$\mathcal{Z}^+(z,t) / u_0$')
plt.colorbar(cp)
# ax.set_xlabel(r'$t / t_0$'))
ax.set_ylabel(r'$z / L_z$')

ax = fig.add_subplot(427)
ax.plot(t, zm_ana(0, t))
ax.set_title(r'$\mathcal{Z}^-(0,t) / u_0$')
ax.set_xlabel(r'$t / t_0$')

ax = fig.add_subplot(428)
cp = ax.contourf(T, Z, zm_ana(Z,T), levels=100)
plt.colorbar(cp)
ax.set_title(r'$\mathcal{Z}^-(z,t) / u_0$')
ax.set_xlabel(r'$t / t_0$')
ax.set_ylabel(r'$z / L_z$')

fig.savefig('temp_figures/leaky_pulse.png', bbox_inches = 'tight', dpi=150)

plt.show(block = False)