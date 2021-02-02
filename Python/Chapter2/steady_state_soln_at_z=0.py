import numpy as np
import matplotlib.pyplot as plt

def abs_u0(omega):
	return u0 / np.sqrt(R ** 2 + 2 * R * np.cos(2 * omega * lz / vA0) + 1)

def abs_b0(omega):
	return B0 * u0 / vA0 / np.sqrt(R ** 2 - 2 * R * np.cos(2 * omega * lz / vA0) + 1)

vA0 = 1
u0 = 1
B0 = 1
lz = 0.5
Lz = 2 * lz
omega_1 = np.pi * vA0 / Lz

omega_min = 0
omega_max = 5 * omega_1
n_omega = 1024
omega = np.linspace(omega_min, omega_max, n_omega)

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = fig_size[1] / 2
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

ax = fig.add_subplot(121)
R = 0.9
ax.plot(omega / omega_1, np.real(abs_u0(omega)), label = 'R = 0.9')
R = 0.5
ax.plot(omega / omega_1, np.real(abs_u0(omega)), label = 'R = 0.5')
R = 0.1
ax.plot(omega / omega_1, np.real(abs_u0(omega)), label = 'R = 0.1')
ax.set_title(r'$|u(0,t)| / u_0$')
ax.set_xlabel(r'$\omega / \omega_1$')
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, bbox_to_anchor=(0.845,1.0))

ax = fig.add_subplot(122)
R = 0.9
ax.plot(omega / omega_1, np.real(abs_b0(omega)))
R = 0.5
ax.plot(omega / omega_1, np.real(abs_b0(omega)))
R = 0.1
ax.plot(omega / omega_1, np.real(abs_b0(omega)))
ax.set_title(r'$|b(0,t)| / b_0$')
ax.set_xlabel(r'$\omega / \omega_1$')
fig.savefig('temp_figures/steady_state_soln_at_z=0.png', bbox_inches = 'tight', dpi=150)

plt.show(block = False)