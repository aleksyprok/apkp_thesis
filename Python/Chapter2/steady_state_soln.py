import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import solve_bvp

def f_driv(t0):
	return u0 * np.exp(1j * omega * t0)

def u_ana(z,t):
	return u0 * np.exp(1j * omega * (t - lz / vA0)) * \
	(np.exp(-1j * omega * z / vA0) - r * np.exp(1j * omega * z / vA0)) / (1 - r ** 2)

def b_ana(z,t):
	return -B0 * u0 / vA0 * np.exp(1j * omega * (t - lz / vA0)) * \
	(np.exp(-1j * omega * z / vA0) + r * np.exp(1j * omega * z / vA0)) / (1 - r ** 2)

def Zp_ana(z,t):
	return u_ana(z,t) + vA0 / B0 * b_ana(z,t)

def Zm_ana(z,t):
	return u_ana(z,t) - vA0 / B0 * b_ana(z,t)


def advection(z, Zpm):
	# Zpm[0,:] = Zr+, Zpm[1,:] = Zi+, Zpm[2,:] = Zr-, Zpm[3,:] = Zi-
	return np.array((-omega / vA0 * Zpm[1], \
					  omega / vA0 * Zpm[0], \
					  omega / vA0 * Zpm[3], \
					 -omega / vA0 * Zpm[2]))

def bcs(Zpm_a, Zpm_b):
	return np.array([Zpm_a[2] - 2 * np.real(f_driv(t0)) + R * Zpm_a[0], \
					 Zpm_a[3] - 2 * np.imag(f_driv(t0)) + R * Zpm_a[1], \
					 Zpm_b[0] + R * Zpm_b[2], \
					 Zpm_b[1] + R * Zpm_b[3]])
vA0 = 1
u0 = 1
R = 0.9
B0 = 1
lz = 0.5
Lz = 2 * lz
t0 = 0

omega_1 = np.pi * vA0 / Lz
omega = (omega_1 + 2 * omega_1) / 2
r = R * np.exp(-2j * omega * lz / vA0)

t_min = 0
t_max = 15
nt = 1024
t = np.linspace(t_min, t_max, nt)

z_min = -lz
z_max = lz
nz = 1024
z = np.linspace(z_min, z_max, nz)

T, Z = np.meshgrid(t,z)

Zpm = np.zeros((4, z.size))
sol = solve_bvp(advection, bcs, z, Zpm)

# Zpm0 = np.array([np.real(Zp_ana(-lz, 0)), \
# 				 np.imag(Zp_ana(-lz, 0)), \
# 				 np.real(Zm_ana(-lz, 0)), \
# 				 np.imag(Zm_ana(-lz, 0))])
# sol = solve_ivp(advection, [z_min, z_max], Zpm0, t_eval = z, rtol = 1e-5, atol = 1e-10)

Zp_num = sol.y[0,:] + 1j * sol.y[1,:]
Zm_num = sol.y[2,:] + 1j * sol.y[3,:]
u_num = (Zp_num + Zm_num) / 2
b_num = (Zp_num - Zm_num) / 2

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = fig_size[1] * 2
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
mk0 = 40

omega = 0
r = R * np.exp(-2j * omega * lz / vA0)
Zpm = np.zeros((4, z.size))
sol = solve_bvp(advection, bcs, z, Zpm)
Zp_num = sol.y[0,:] + 1j * sol.y[1,:]
Zm_num = sol.y[2,:] + 1j * sol.y[3,:]
u_num = (Zp_num + Zm_num) / 2
b_num = (Zp_num - Zm_num) / 2

ax = fig.add_subplot(421)
ax.plot(z, np.real(u_ana(z, 0)), color = 'xkcd:blue', label = 'Real part')
ax.plot(z, np.imag(u_ana(z, 0)), color = 'xkcd:orange', label = 'Imaginary part')
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='center')
ax.plot(z, np.real(u_num), '+', color = 'xkcd:blue', markevery = mk0)
ax.plot(z, np.imag(u_num), '+', color = 'xkcd:orange', markevery = mk0)
ax.set_title(r'$u(z,0) / u_0$')
ax.text(0.85, 1.075, \
	r'$\omega = 0,\  R = $' + '{:1.2f}'.format(R),
	fontsize = 12, \
	transform=ax.transAxes)

ax = fig.add_subplot(422)
ax.plot([], [], color = 'k', label = 'Analytic solution')
ax.plot([], [], '+', color = 'k', label = 'Numerical solution')
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='center')
ax.plot(z, np.real(b_ana(z, 0)), color = 'xkcd:blue')
ax.plot(z, np.real(b_num), '+', color = 'xkcd:blue', markevery = mk0)
ax.plot(z, np.imag(b_ana(z, 0)), color = 'xkcd:orange')
ax.plot(z, np.imag(b_num), '+', color = 'xkcd:orange', markevery = mk0)
ax.set_title(r'$b(z,0) / b_0$')

omega = omega_1 / 2
r = R * np.exp(-2j * omega * lz / vA0)
Zpm = np.zeros((4, z.size))
sol = solve_bvp(advection, bcs, z, Zpm)
Zp_num = sol.y[0,:] + 1j * sol.y[1,:]
Zm_num = sol.y[2,:] + 1j * sol.y[3,:]
u_num = (Zp_num + Zm_num) / 2
b_num = (Zp_num - Zm_num) / 2

ax = fig.add_subplot(423)
ax.plot(z, np.real(u_ana(z, 0)), color = 'xkcd:blue')
ax.plot(z, np.real(u_num), '+', color = 'xkcd:blue', markevery = mk0)
ax.plot(z, np.imag(u_ana(z, 0)), color = 'xkcd:orange')
ax.plot(z, np.imag(u_num), '+', color = 'xkcd:orange', markevery = mk0)
ax.text(1, 1.075, \
	r'$\omega = \frac{1}{2}\omega_1$', \
	fontsize = 12, \
	transform=ax.transAxes)


ax = fig.add_subplot(424)
ax.plot(z, np.real(b_ana(z, 0)), color = 'xkcd:blue')
ax.plot(z, np.real(b_num), '+', color = 'xkcd:blue', markevery = mk0)
ax.plot(z, np.imag(b_ana(z, 0)), color = 'xkcd:orange')
ax.plot(z, np.imag(b_num), '+', color = 'xkcd:orange', markevery = mk0)

omega = omega_1
r = R * np.exp(-2j * omega * lz / vA0)
Zpm = np.zeros((4, z.size))
sol = solve_bvp(advection, bcs, z, Zpm)
Zp_num = sol.y[0,:] + 1j * sol.y[1,:]
Zm_num = sol.y[2,:] + 1j * sol.y[3,:]
u_num = (Zp_num + Zm_num) / 2
b_num = (Zp_num - Zm_num) / 2

ax = fig.add_subplot(425)
ax.plot(z, np.real(u_ana(z, 0)), color = 'xkcd:blue')
ax.plot(z, np.real(u_num), '+', color = 'xkcd:blue', markevery = mk0)
ax.plot(z, np.imag(u_ana(z, 0)), color = 'xkcd:orange')
ax.plot(z, np.imag(u_num), '+', color = 'xkcd:orange', markevery = mk0)
ax.text(1.025, 1.075, \
	r'$\omega = \omega_1$', \
	fontsize = 12, \
	transform=ax.transAxes)

ax = fig.add_subplot(426)
ax.plot(z, np.real(b_ana(z, 0)), color = 'xkcd:blue')
ax.plot(z, np.real(b_num), '+', color = 'xkcd:blue', markevery = mk0)
ax.plot(z, np.imag(b_ana(z, 0)), color = 'xkcd:orange')
ax.plot(z, np.imag(b_num), '+', color = 'xkcd:orange', markevery = mk0)

epsilon = 0.1
omega = 2 * omega_1 + epsilon * omega_1
r = R * np.exp(-2j * omega * lz / vA0)
Zpm = np.zeros((4, z.size))
sol = solve_bvp(advection, bcs, z, Zpm)
Zp_num = sol.y[0,:] + 1j * sol.y[1,:]
Zm_num = sol.y[2,:] + 1j * sol.y[3,:]
u_num = (Zp_num + Zm_num) / 2
b_num = (Zp_num - Zm_num) / 2

ax = fig.add_subplot(427)
ax.plot(z, np.real(u_ana(z, 0)), color = 'xkcd:blue')
ax.plot(z, np.real(u_num), '+', color = 'xkcd:blue', markevery = mk0)
ax.plot(z, np.imag(u_ana(z, 0)), color = 'xkcd:orange')
ax.plot(z, np.imag(u_num), '+', color = 'xkcd:orange', markevery = mk0)
ax.set_xlabel(r'$z / L_z$')
ax.text(0.75, 1.075, \
	r'$\omega = \omega_2 + \epsilon\omega_1,\  \epsilon = $' + '{:1.2f}'.format(epsilon), \
	fontsize = 12, \
	transform=ax.transAxes)

ax = fig.add_subplot(428)
ax.plot(z, np.real(b_ana(z, 0)), color = 'xkcd:blue')
ax.plot(z, np.real(b_num), '+', color = 'xkcd:blue', markevery = mk0)
ax.plot(z, np.imag(b_ana(z, 0)), color = 'xkcd:orange')
ax.plot(z, np.imag(b_num), '+', color = 'xkcd:orange', markevery = mk0)
ax.set_xlabel(r'$z / L_z$')


fig.savefig('temp_figures/steady_state_soln_along_z.png', bbox_inches = 'tight', dpi=150)


plt.show(block = False)