import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def white_noise(t):
	eta = a[0] * np.random.normal()
	for k in range(1,N+1):
		eta += a[k] * np.random.normal() * np.cos(omega[k] * t)
		eta += b[k] * np.random.normal() * np.sin(omega[k] * t)
	return eta

def red_noise(y, t):
	return integrate.cumtrapz(y, t, initial=0)

def variance_exact(t):
	return D / 2 / omega_n ** 3 * (omega_n * t - np.sin(omega_n * t) * np.cos(omega_n * t))

def diff_variance_exact(t):
	return D / 2 / omega_n      * (omega_n * t + np.sin(omega_n * t) * np.cos(omega_n * t))

def variance_fourier(t):
	var_for = variance_fourier_term1(t, 0)
	for k in range(1,N+1):
		if omega[k] != omega_n:
			var_for += variance_fourier_term1(t, k)
		elif omega[k] == omega_n:
			var_for += variance_fourier_term2(t, k)
	return var_for

def variance_fourier_term1(t, k):
	term1  = a[k] ** 2 * omega_n ** 2 * (np.cos(omega_n * t) - np.cos(omega[k] * t)) ** 2
	term1 += b[k] ** 2 * (omega[k] * np.sin(omega_n * t) - omega_n * np.sin(omega[k] * t)) ** 2
	term1 = term1 / omega_n ** 2 / (omega[k] ** 2 - omega_n ** 2) ** 2
	return term1

def variance_fourier_term2(t, k):
	term2  = a[k] ** 2 * omega_n ** 2 * t ** 2 * np.sin(omega_n * t) ** 2
	term2 += b[k] ** 2 * (np.sin(omega_n * t) - omega_n * t * np.cos(omega_n * t)) ** 2
	term2 = term2 / 4 / omega_n ** 4
	return term2

def diff_variance_fourier(t):
	var_for = diff_variance_fourier_term1(t, 0)
	for k in range(1,N+1):
		if omega[k] != omega_n:
			var_for += diff_variance_fourier_term1(t, k)
		elif omega[k] == omega_n:
			var_for += diff_variance_fourier_term2(t, k)
	return var_for

def diff_variance_fourier_term1(t, k):
	term1  = a[k] ** 2 * (omega[k] * np.sin(omega[k] * t) - omega_n * np.sin(omega_n * t)) ** 2
	term1 += b[k] ** 2 * omega[k] ** 2 * (np.cos(omega_n * t) - np.cos(omega[k] * t)) ** 2
	term1 = term1 / (omega[k] ** 2 - omega_n ** 2) ** 2
	return term1

def diff_variance_fourier_term2(t, k):
	term2  = a[k] ** 2 * (np.sin(omega_n * t) + omega_n * t * np.cos(omega_n * t)) ** 2
	term2 += b[k] ** 2 * omega_n ** 2 * t ** 2 * np.sin(omega_n * t) ** 2
	term2 = term2 / 4 / omega_n ** 2
	return term2

omega_n = np.pi
D = 1
nt = 1024
t_end = 10 / omega_n
t = np.linspace(0, t_end, nt)

P = t_end
N = 1000
k_array = np.arange(N+1)
omega = np.pi * k_array / (2 * P)

a = np.full(N+1, np.sqrt(D / P / 2))
b = np.full(N+1, np.sqrt(D / P / 2))
a[0] = np.sqrt(D / P) / 2
b[0] = 0

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = 1.75 * fig_size[1]
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

ax = fig.add_subplot(321)
ax.plot(t * omega_n, variance_exact(t)/ (D / omega_n ** 3), label = 'Exact')
ax.plot(t * omega_n, variance_fourier(t)/ (D / omega_n ** 3), '-.', label = 'Approximation')
ax.set_xlabel(r'$\omega_n t$')
ax.set_title(r'$\left\langle y_n^2(t)\right\rangle \, /\, y_{n0}^2$')
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, bbox_to_anchor=(1.475,1.35))
ax.text(0.1, 0.75, \
	r'$P = $' + '{:1.0f}'.format(P * omega_n ) + r'$\, /\, \omega_n$' + '\n' + \
	r'$N = $' + '{:d}'.format(N), \
	transform=ax.transAxes)
ax = fig.add_subplot(322)
ax.plot(t * omega_n, diff_variance_exact(t) / (D / omega_n))
ax.plot(t * omega_n, diff_variance_fourier(t) / (D / omega_n), '-.')
ax.set_xlabel(r'$\omega_n t$')
ax.set_title(r'$\left\langle \dot{y}_n^2(t)\right\rangle \, /\, \dot{y}_{n0}^2$')
ax.text(0.1, 0.7, \
	r'$y_{n0}^2 = D \, / \, \omega_n^3$' '\n' + \
	r'$\dot{y}_{n0}^2 = D \, / \, \omega_n$', \
	transform=ax.transAxes)

N_min = 0
N_max = 3
nN = 128
N_array = np.int_(np.round(np.logspace(N_min, N_max, nN)))

err_array = np.zeros(nN)
diff_err_array = np.zeros(nN)

for iN in range(nN):
	N = N_array[iN]
	k_array = np.arange(N+1)
	omega = np.pi * k_array / (2 * P)
	a = np.full(N+1, np.sqrt(D / P / 2))
	b = np.full(N+1, np.sqrt(D / P / 2))
	a[0] = np.sqrt(D / P) / 2
	b[0] = 0
	err_array[iN] = variance_exact(t_end) - variance_fourier(t_end)
	diff_err_array[iN] = diff_variance_exact(t_end) - diff_variance_fourier(t_end)

ax = fig.add_subplot(323)
ax.loglog(N_array, err_array / variance_exact(t_end))
ax.set_title(r'$\left[\left\langle y_n^2(P)\right\rangle - \left\langle y_{n,N}^2(P)\right\rangle\right]\, \left/\, \left\langle y_n^2(P)\right\rangle\right.$')
ax.set_xlabel(r'$N$')

ax = fig.add_subplot(324)
ax.loglog(N_array, diff_err_array / diff_variance_exact(t_end))
ax.set_title(r'$\left[\left\langle \dot{y}_n^2(P)\right\rangle - \left\langle \dot{y}_{n,N}^2(P)\right\rangle\right]\, \left/\, \left\langle \dot{y}_n^2(P)\right\rangle\right.$')
ax.set_xlabel(r'$N$')

white_noise_array = white_noise(t)
ax = fig.add_subplot(326)
ax.plot(t * omega_n, white_noise_array)
ax.set_xlabel(r'$\omega_n t$')
ax.set_title('White noise')

ax = fig.add_subplot(325)
ax.plot(t * omega_n, red_noise(white_noise_array, t))
ax.set_xlabel(r'$\omega_n t$')
ax.set_title('Red noise')

fig.savefig('temp_figures/noisy_driver.pdf', bbox_inches = 'tight')

plt.show(block = False)	

# P_min = 1
# P_max = 3
# n_P = 16
# P_array = np.logspace(P_min, P_max, n_P)

# N_min = 1
# N_max = 4
# nN = 16
# N_array = np.int_(np.round(np.logspace(N_min, N_max, nN)))

# NN, PP = np.meshgrid(N_array, P_array)

# err_array = np.zeros((n_P, nN), dtype = float)
# for iN in range(nN):
# 	print(iN)
# 	for iP in range(n_P):
# 		x=1
# 		N = N_array[iN]
# 		P = P_array[iP]
# 		k_array = np.arange(N+1)
# 		omega = np.pi * k_array / (2 * P)
# 		a = np.full(N+1, np.sqrt(D / P / 2))
# 		b = np.full(N+1, np.sqrt(D / P / 2))
# 		a[0] = np.sqrt(D / P) / 2
# 		b[0] = 0

# 		err_array[iP,iN] = np.abs(variance_exact(t_end) - variance_fourier(t_end))