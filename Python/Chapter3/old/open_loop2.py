import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fftpack import diff as psdiff

def vA(x):
	return vA0 * (1 + x / a)

def vAx(x):
	return vA0 / a

def h(x):
	# h0 = np.zeros_like(x)
	# h0 += np.sin(np.pi * x / lx) ** 2 * (np.abs(x) >  lx / 2)
	# h0 += f0                          * (np.abs(x) <= lx / 2)
	# return h0
	return np.full_like(x, f0)

# def dZmdz(z, Zm):

# 	Zm_xx = np.roll(Zm, -1) - 2 * Zm + np.roll(Zm, 1)
# 	Zm_xx = Zm_xx / dx ** 2
# 	# Zm_xx = psdiff(Zm, order = 2, period = 2 * lx)

# 	Zm_z = (-1j * omega * Zm + nu_p * Zm_xx) / vA(x)
# 	Zm_z[0]  = 0
# 	Zm_z[-1] = 0

# 	return Zm_z

def dZmdz(z, Zm):

	Zm_xx = np.roll(Zm, -1) - 2 * Zm + np.roll(Zm, 1)
	Zm_xx[0]  = -2 * Zm[0]  + 2 * Zm[1]
	Zm_xx[-1] =  2 * Zm[-2] - 2 * Zm[-1]
	Zm_xx = Zm_xx / dx ** 2

	Zm_z = (-1j * omega * Zm + nu_p * Zm_xx) / vA(x)

	return Zm_z

vA0 = 1
B0 = 1
f0 = 1
omega = np.pi
a = 1

eta = 0.25e-3
nu = 0.75e-3
nu_p = 0.5 * (eta + nu)

kx = (6 * omega / (eta + nu) * a) ** (1/3)
lx = 5 / kx
x_min = -lx
x_max = lx
nx = 512
dx = (x_max - x_min) / (nx - 1)
x = np.linspace(x_min, x_max, nx)

ix0 = nx // 2
x0 = x[ix0]
# Lph = (vAx(x0) ** 2 * (eta + nu) / (6 * vA(x0) ** 5) * omega ** 2) ** (-1/3)
kz0 = omega / vA0
Lph = (6 * omega * a ** 2 / (eta + nu)) ** (1/3) / kz0

z_min = 0
z_max = 2 * Lph
nz = 512
dz = (z_max - z_min) / (nz - 1)
z = np.linspace(z_min, z_max, nz)

X, Z = np.meshgrid(x,z)


Zm0 = 2 * h(x) + 0j

sol = solve_ivp(dZmdz, [z_min, z_max], Zm0, t_eval = z)
Zm = sol.y.T
u =  0.5 * Zm
b = -0.5 * Zm * B0 / vA(X)

ix0 = nx // 2
iz0 = nz // 2

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(x, h(x))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, u[:,ix0].real)
ax.plot(z, u[:,ix0].imag)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, u[iz0,:].real)
ax.plot(x, u[iz0,:].imag)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, u[-1,:].real)
ax.plot(x, u[-1,:].imag)

fig = plt.figure()
ax = fig.add_subplot(111)
cp = ax.contourf(X, Z, u.real, levels=100)

plt.show(block = False)