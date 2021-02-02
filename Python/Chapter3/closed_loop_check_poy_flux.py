import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_bvp
from scipy.integrate import trapz

def vA(x):
	return vA0 * (1 + x / a0)

def vAx(x):
	return vA0 / a0

def rho(x):
	return B0 ** 2 / vA(x) ** 2

def a(x):
	return vA(x) / vAx(x)


def dYdx(x, Y):

	y = Y[0] + 1j * Y[1]
	v = Y[2] + 1j * Y[3]

	dy = v
	dv = 1j * (omega ** 2 - vA(x) ** 2 * kzn ** 2) * y / (omega * eta) \
	   + 2j * omega * f0 / (n * np.pi * eta)

	return np.array([dy.real, dy.imag, dv.real, dv.imag])

def bcs(Y_a, Y_b):
	return np.array([Y_a[2], Y_a[3], Y_b[2], Y_b[3]])

def kz(x):
	return omega / vA(x)

def Lph(x):
	return (6 * omega * a(x) ** 2 / eta) ** (1/3) / kz(x)

def A_infty(x,z):
	A_infty = x - x + z - z + 0j
	for k in range(Ns):
		if k % 100 == 0: print(k)
		zk = (-1) ** k * (z - lz) + (2 * k + 1) * lz + 0j
		A_infty += (-1) ** k * np.exp(-(zk / Lph(x)) ** 3 - 1j * kz(x) * zk)
	return A_infty

def u_ana(x,z):
	return f0 * A_infty(x,z)

def gamma_approx(x):
	return vA(x) / Lph(x)

vA0 = 1
B0 = 1
f0 = 1
Lz = 1
lz = Lz / 2
a0 = 1
omega = np.pi * vA0 / Lz
eta = 1e-4
Nh = 10
Ns = int(round(10 * Lph(0) / lz))

kx = (6 * omega / (eta * a0)) ** (1/3)
lx = min(10 / kx, 0.5)
x_min = -lx
x_max = lx
nx = 512
dx = (x_max - x_min) / nx
x = np.linspace(x_min + dx / 2, x_max - dx / 2, nx)

z_min = 0
z_max = Lz
nz = 128
dz = (z_max - z_min) / (nz - 1)
z = np.linspace(z_min, z_max, nz)

X, Z = np.meshgrid(x,z)
ix0 = nx // 2
iz0 = nz // 2
x0 = x[ix0]
z0 = z[iz0]

y  = np.zeros((nz, nx), dtype = complex)
yx = np.zeros((nz, nx), dtype = complex)
b  = np.zeros((nz, nx), dtype = complex)
bx = np.zeros((nz, nx), dtype = complex)
x_temp = np.linspace(x_min, x_max, 5)
Y = np.zeros((4, x_temp.size))
Y[0, 2] = 1
for n in range(1,Nh+1):
	kzn = n * np.pi / Lz
	sol = solve_bvp(dYdx, bcs, x_temp, Y)
	y  += (sol.sol(X)[0] + 1j * sol.sol(X)[1]) * np.sin(kzn * Z)
	yx += (sol.sol(X)[2] + 1j * sol.sol(X)[3]) * np.sin(kzn * Z)
	b  += kzn * (sol.sol(X)[0] + 1j * sol.sol(X)[1]) * np.cos(kzn * Z)
	bx += kzn * (sol.sol(X)[2] + 1j * sol.sol(X)[3]) * np.cos(kzn * Z)
u = y + f0 * (Lz - Z) / Lz
ux = yx
b  = B0 / (1j * omega) * (b - f0 / Lz)
bx = B0 / (1j * omega) * bx

poy_flux = -0.5 * B0 * np.real(u[0,:] * b[0,:].conjugate())
ohmic_heating = 0.25 * eta * trapz(np.abs(ux) ** 2 + np.abs(bx) ** 2, x = z, axis = 0)
avg_engy = 0.25 * trapz(np.abs(u) ** 2 + np.abs(b) ** 2, x = z, axis = 0)
# ohmic_heating = 0.25 * eta * np.abs(bx[0,:]) ** 2
# avg_engy = 0.25 * np.abs(b[0,:]) ** 2

poy_flux_total = trapz(poy_flux, x = x)
ohmic_total = trapz(ohmic_heating, x = x)
engy_total = trapz(avg_engy, x = x)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, poy_flux)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, ohmic_heating)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, ohmic_heating / avg_engy)
ax.plot(x, gamma_approx(x))

print(poy_flux_total)
print(ohmic_total)
print(ohmic_total / engy_total)


plt.show(block = False)