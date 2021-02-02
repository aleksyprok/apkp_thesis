import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fftpack import diff as psdiff

def vA(x):
	# return vA0 * (1 + a / np.pi * np.arcsin(np.sin(np.pi / Lx * x)))
	return vA0 * (1 + a * np.sin(np.pi / Lx * x))

def dUdz(z, U):

	u = U[0:nx]
	b = U[nx:]

	uxx = psdiff(u, order = 2, period = 2 * Lx)
	bxx = psdiff(b, order = 2, period = 2 * Lx)

	# uz =  1j * omega * b - eta * bxx
	# bz = (1j * omega * u - nu  * uxx) / vA(x) ** 2
	uz =  1j * omega * b + eta * uxx
	bz = (1j * omega * u + nu  * bxx) / vA(x) ** 2
	return np.concatenate((uz, bz))

Lx = 1
vA0 = 1
f0 = 1
omega = np.pi
Lz = 2 * np.pi * vA0 / omega
a = 0.5

eta = 1e-3
nu = 0

x_min = -Lx
x_max = Lx
nx = 128
dx = (x_max - x_min) / nx
x = np.linspace(x_min + dx / 2, x_max - dx / 2, nx)

z_min = 0
z_max = 10 * Lz
nz = 256
dz = (z_max - z_min) / (nz - 1)
z = np.linspace(z_min, z_max, nz)

u0 = np.full(nx,  f0, dtype = complex)
b0 = np.full(nx, -f0, dtype = complex)
U0 = np.concatenate((u0, b0))
sol = solve_ivp(dUdz, [z_min, z_max], U0, t_eval = z)
u = sol.y[0:nx,:].T
b = sol.y[nx:,:].T

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, np.real(u[:,64]))
ax.plot(z, np.imag(u[:,64]))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, np.real(b[:,64]))
ax.plot(z, np.imag(b[:,64]))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(x, np.real(u[64,:]))
# ax.plot(x, np.imag(u[64,:]))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(x, np.real(b[64,:]))
# ax.plot(x, np.imag(b[64,:]))

plt.show(block = False)
