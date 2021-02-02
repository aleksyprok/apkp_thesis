import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fftpack import diff as psdiff

def vA(x):
	return vA0 * (1 + a * np.sin(np.pi / Lx * x))

def vAx(x):
	return np.pi * vA0 * a / Lx * np.cos(np.pi / Lx * x)

def kz(x):
	return omega / vA(x)

def Lph(x):
	return (vAx(x) ** 2 * (eta + nu) / (6 * vA(x) ** 5) * omega ** 2) ** (-1/3)

def dZdz(z, Z):

	Zp = Z[0:nx]
	Zm = Z[nx:]

	Zp_xx = psdiff(Zp, order = 2, period = 2 * Lx)
	Zm_xx = psdiff(Zm, order = 2, period = 2 * Lx)

	Zp_z = ( 1j * omega * Zp + nu_p * Zp_xx + nu_m * Zm_xx) / vA(x)
	Zm_z = (-1j * omega * Zm + nu_p * Zm_xx + nu_m * Zp_xx) / vA(x)

	return np.concatenate((Zp_z, Zm_z))

def A_infty(x,z):
	A_infty = x - x + z - z + 0j
	for k in range(N):
		zk = (-1) ** k * z + (2 * k + 1) * lz + 0j
		A_infty += (-1) ** k * np.exp(-(zk / Lph(x)) ** 3 - 1j * kz(x) * zk)
	return A_infty

def B_infty(x,z):
	B_infty = x - x + z - z + 0j
	for k in range(N):
		zk = (-1) ** k * z + (2 * k + 1) * lz + 0j
		B_infty -= np.exp(-(zk / Lph(x)) ** 3 - 1j * kz(x) * zk)
	return B_infty

def u_ana(x,z):
	return f0 * A_infty(x,z)

def b_ana(x,z):
	return B0 * f0 / vA(x) * B_infty(x,z)



vA0 = 1
B0 = 1
f0 = 1
lz = 0.5
Lz = 2 * lz
omega = np.pi * vA0 / Lz
Lx = Lz
a = 0.25

eta = 0.25e-4
nu = 0.75e-4
nu_p = 0.5 * (eta + nu)
nu_m = 0.5 * (eta - nu)

x_min = -Lx
x_max = Lx
nx = 512
dx = (x_max - x_min) / nx
x = np.linspace(x_min + dx / 2, x_max - dx / 2, nx)

ix0 = nx // 2
x0 = x[ix0]
Lph0 = (vAx(x0) ** 2 * (eta + nu) / (6 * vA(x0) ** 5) * omega ** 2) ** (-1/3)

N = int(round(10 * Lph0 / lz))

z_min = -lz
z_max = lz
nz = 512
dz = (z_max - z_min) / (nz - 1)
z = np.linspace(z_min, z_max, nz)

X, Z = np.meshgrid(x,z)

Zp0 = u_ana(x,-lz) + B0 * f0 / vA(x) * b_ana(x,-lz)
Zm0 = u_ana(x,-lz) - B0 * f0 / vA(x) * b_ana(x,-lz)
Z0 = np.concatenate((Zp0, Zm0))

sol = solve_ivp(dZdz, [z_min, z_max], Z0, t_eval = z, rtol = 1e-4, atol = 1e-8)
Zp = sol.y[0:nx,:].T
Zm = sol.y[nx:,:].T
u = 0.5 * (Zp + Zm)
b = 0.5 * (Zp - Zm) * B0 / vA(X)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, np.real(u[:,0]))
ax.plot(z, np.real(u_ana(0,z)))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, np.imag(u[:,0]))
ax.plot(z, np.imag(u_ana(0,z)))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(z, np.real(b[:,0]))
# ax.plot(z, np.imag(b[:,0]))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(z, np.real(u_ana(0,z)))
# ax.plot(z, np.imag(u_ana(0,z)))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(z, np.real(u_ana(x,0)))
# ax.plot(z, np.imag(u_ana(x,0)))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# cp = ax.contourf(X, Z, np.real(u_ana(X,Z)), levels=100)
# plt.colorbar(cp)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# cp = ax.contourf(X, Z, np.imag(u_ana(X,Z)), levels=100)
# plt.colorbar(cp)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(z, np.real(b_ana(0,z)))
# ax.plot(z, np.imag(b_ana(0,z)))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(z, np.real(b_ana(x,0)))
# ax.plot(z, np.imag(b_ana(x,0)))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# cp = ax.contourf(X, Z, np.real(b_ana(X,Z)), levels=100)
# plt.colorbar(cp)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# cp = ax.contourf(X, Z, np.imag(b_ana(X,Z)), levels=100)
# plt.colorbar(cp)

plt.show(block = False)