import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fftpack import diff as psdiff

def vA(x):
	return vA0 * (1 + 1 / np.pi * np.arcsin(np.sin(np.pi / Lx * x)))

def dudz(z, u):
	uxx = psdiff(u, order = 2, period = 2 * Lx)
	return (-1j * omega * u + 0.5 * (eta + nu) * uxx) / vA(x)

Lx = 1
vA0 = 1
f0 = 1
omega = np.pi
Lz = 2 * np.pi * vA0 / omega
a = 1

eta = 1e-3
nu = 1e-3

x_min = -Lx
x_max = Lx
nx = 128
dx = (x_max - x_min) / nx
x = np.linspace(x_min + dx / 2, x_max - dx / 2, nx)

z_min = 0
z_max = 5 * Lz
nz = 256
dz = (z_max - z_min) / (nz - 1)
z = np.linspace(z_min, z_max, nz)

u0 = np.full(nx,  f0, dtype = complex)
sol = solve_ivp(dudz, [z_min, z_max], u0, t_eval = z)
u = sol.y.T


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, np.real(u[:,64]))
ax.plot(z, np.imag(u[:,64]))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(x, psdiff(np.sin(np.pi * x / Lx), order = 2, period = 2 * Lx))

plt.show(block = False)