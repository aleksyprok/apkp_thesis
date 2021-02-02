import numpy as np
import matplotlib.pyplot as plt

def white_noise(t):
	eta = a[0] * np.random.normal()
	for k in range(1,N+1):
		eta += a[k] * np.random.normal() * np.cos(omega[k] * t)
		eta += b[k] * np.random.normal() * np.sin(omega[k] * t)
	return eta

def variance_exact(t):
	return E / 2 / omega_n * (omega_n * t + np.sin(omega_n * t) * np.cos(omega_n * t))

def variance_fourier(t):
	var_for = variance_fourier_term1(t, 0)
	for k in range(1,N+1):
		if omega[k] != omega_n:
			var_for += variance_fourier_term1(t, k)
		elif omega[k] == omega_n:
			var_for += variance_fourier_term2(t, k)
	return var_for

def variance_fourier_term1(t, k):
	term1  = a[k] ** 2 * (omega[k] * np.sin(omega[k] * t) - omega_n * np.sin(omega_n * t)) ** 2
	term1 += b[k] ** 2 * omega[k] ** 2 * (np.cos(omega_n * t) - np.cos(omega[k] * t)) ** 2
	term1 = term1 / (omega[k] ** 2 - omega_n ** 2) ** 2
	return term1

def variance_fourier_term2(t, k):
	term2  = a[k] ** 2 * (np.sin(omega_n * t) + omega_n * t * np.cos(omega_n * t)) ** 2
	term2 += b[k] ** 2 * omega_n ** 2 * t ** 2 * np.sin(omega_n * t) ** 2
	term2 = term2 / 4 / omega_n ** 2
	return term2


P = 10
N = 1000
k_array = np.arange(N+1)
omega = np.pi * k_array / (2 * P)
xi = k_array / (4 * P)
omega_n = np.pi
E = 1

t_min = 0
t_max = min(P, 10)
nt = 1024
t = np.linspace(t_min, t_max, nt)

a = np.full(N+1, np.sqrt(E / P / 2))
b = np.full(N+1, np.sqrt(E / P / 2))
a[0] = np.sqrt(E / P) / 2
b[0] = 0

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[0] = 2 * fig_size[0]
fig.set_size_inches(fig_size)

ax = fig.add_subplot(121)
ax.plot(t, variance_exact(t))
ax.plot(t, variance_fourier(t))
ax.set_xlabel('t')
ax.set_title(r'$\langle x^2(t)\rangle$ - Exact')

ax = fig.add_subplot(122)
ax.plot(t, variance_fourier(t))
ax.set_xlabel('t')
ax.set_title(r'$\langle x^2(t)\rangle$ - Fourier')

plt.show(block = False)	