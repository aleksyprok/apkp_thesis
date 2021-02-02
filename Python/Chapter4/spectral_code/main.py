import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from scipy.integrate import solve_ivp
from scipy.integrate import solve_bvp
from scipy.integrate import quad
import control as ct
import eigens as eg

def ux_ana(x,z):
	return -beta0 * np.log(x - 1j * xi) * ( \
			1j * ct.k_perp * eg.phi(z,k) - np.sin(ct.alpha) * eg.phi_prime(z,k))

def b_par_ana(z):
	return Y_sol.sol(z)[0] + 1j * Y_sol.sol(z)[1]

def u_perp_ana(x,z):
	return beta0 * eg.phi(z,k) / (x - 1j * xi)

def nabla_perp_eqn(z, Y):
	b_par = Y[0] + 1j * Y[1]
	db_par = 1j * ct.k_perp * b_par / np.sin(ct.alpha) - \
		   	 2j * beta0 * omega_r / (ct.a0 * np.sin(ct.alpha)) \
		   * eg.phi(z,k) / ct.vA(0,z) ** 2
	return np.array([db_par.real, db_par.imag])

def bcs(Y_a, Y_b):
	return np.array([Y_a[0] - Y_b[0], Y_a[1] - Y_b[1]])

def dU(x, U):

	ux1 = U[0:ct.N+1]
	ux2 = U[ct.N+1:2*(ct.N+1)]
	b_par1 = U[2*(ct.N+1):3*(ct.N+1)]
	b_par2 = U[3*(ct.N+1):]

	Ux1 = np.tile(np.array([ux1]).T, (1, ct.N+1))
	Ux2 = np.tile(np.array([ux2]).T, (1, ct.N+1))
	B_par1 = np.tile(np.array([b_par1]).T, (1, ct.N+1))
	B_par2 = np.tile(np.array([b_par2]).T, (1, ct.N+1))

	delta_perp1 = -1j * omega / (omega ** 2 / ct.vAx(x) ** 2 - ct.omega_n ** 2) * ( \
		np.tan(ct.alpha) ** 2 * ct.omega_n ** 2 * b_par1 + np.sum( \
		ct.k_perp ** 2 * B_par1 * ct.I3 + \
		2j * ct.k_perp * np.sin(ct.alpha) * B_par2 * ct.I5, axis = 0))

	delta_perp2 = -1j * omega / (omega ** 2 / ct.vAx(x) ** 2 - ct.varpi_n ** 2) * ( \
		np.tan(ct.alpha) ** 2 * ct.varpi_n ** 2 * b_par2 + np.sum( \
		ct.k_perp ** 2 * B_par2 * ct.I4 + \
		2j * ct.k_perp * np.sin(ct.alpha) * B_par1 * ct.I6, axis = 0))

	dux1 = -(1j * omega * b_par1 + delta_perp1)

	dux2 = -(1j * omega * b_par2 + delta_perp2)

	db_par1 = -1j / omega * np.sum( \
		(omega ** 2 / ct.vAx(x) ** 2 - ct.Omega_n ** 2) * Ux1 * ct.I1, axis = 0)

	db_par2 = -1j / omega * np.sum( \
		(omega ** 2 / ct.vAx(x) ** 2 - ct.Varpi_n ** 2) * Ux2 * ct.I2, axis = 0)

	return np.concatenate((dux1, dux2, db_par1, db_par2))

eg.calc_eigenfrequencies()
eg.get_parameters()
eg.calc_inner_products()

omega_r = np.pi * ct.vAp * np.cos(ct.alpha) / ct.Lz
omega_i = 1e-5 * omega_r
omega = omega_r + 1j * omega_i

xi = ct.a0 * omega_i / omega_r

nx = 256
lx = 1 * abs(xi)
x_min = -8 * lx
x_max =  8 * lx
dx = (x_max - x_min) / (nx - 1)
x = np.linspace(x_min, x_max, nx)

nz = 1024
z_min = -ct.Lz
z_max =  ct.Lz
dz = (z_max - z_min) / nz
z = np.linspace(z_min, z_max, nz)

X, Z = np.meshgrid(x, z)

ux10 = np.zeros(ct.N+1, dtype = complex)
ux20 = np.zeros(ct.N+1, dtype = complex)
b_par10 = np.zeros(ct.N+1, dtype = complex)
b_par20 = np.zeros(ct.N+1, dtype = complex)

# Excite the kth resonance
k = 3
beta0 = 1

ux10[k] = -1j * ct.k_perp * beta0 * np.log(x_min - 1j * xi)
ux20 = beta0 * np.sin(ct.alpha) * np.log(x_min - 1j * xi) * ct.I7[k,:]

# Calc b_par0
z_nodes = np.linspace(z_min, z_max, 1000)
Y_nodes = np.zeros((2, z_nodes.size))
Y_sol = solve_bvp(nabla_perp_eqn, bcs, z_nodes, Y_nodes, \
				  tol = ct.errtol, max_nodes = 10000, verbose = 2)
integrand1_r = lambda z, n: b_par_ana(z).real *    eg.phi(z,n) / ct.vA(0,z) ** 2
integrand1_i = lambda z, n: b_par_ana(z).imag *    eg.phi(z,n) / ct.vA(0,z) ** 2
integrand2_r = lambda z, n: b_par_ana(z).real * eg.varphi(z,n) / ct.vA(0,z) ** 2
integrand2_i = lambda z, n: b_par_ana(z).imag * eg.varphi(z,n) / ct.vA(0,z) ** 2
for n in range(ct.N+1):
	if n == 0: print('Calculating b_par...')
	if n % 10 == 0: print(n)
	int1_r, w1 = quad(integrand1_r, z_min, z_max, args = (n), limit = 1000, \
						epsabs = ct.errtol, epsrel = ct.errtol)
	int1_i, w1 = quad(integrand1_i, z_min, z_max, args = (n), limit = 1000, \
						epsabs = ct.errtol, epsrel = ct.errtol)
	int2_r, w1 = quad(integrand2_r, z_min, z_max, args = (n), limit = 1000, \
						epsabs = ct.errtol, epsrel = ct.errtol)
	int2_i, w1 = quad(integrand2_i, z_min, z_max, args = (n), limit = 1000, \
						epsabs = ct.errtol, epsrel = ct.errtol)
	b_par10[n] = int1_r + 1j * int1_i
	b_par20[n] = int2_r + 1j * int2_i

U0 = np.concatenate((ux10, ux20, b_par10, b_par20))

sol = solve_ivp(dU, [x_min, x_max], U0, t_eval = x, method = 'RK45', \
				rtol = ct.errtol, atol = ct.errtol)

ux1n = sol.y[0:ct.N+1]
ux2n = sol.y[ct.N+1:2*(ct.N+1)]
b_par1n = sol.y[2*(ct.N+1):3*(ct.N+1)]
b_par2n = sol.y[3*(ct.N+1):]

# Calculate u_perpn
u_perp1n = np.zeros((ct.N+1,nx), dtype =  complex)
u_perp2n = np.zeros((ct.N+1,nx), dtype =  complex)
for ix in range(nx):
	B_par1n = np.tile(np.array([b_par1n[:,ix]]).T, (1, ct.N+1))
	B_par2n = np.tile(np.array([b_par2n[:,ix]]).T, (1, ct.N+1))
	u_perp1n[:,ix] = 1j * omega / (omega ** 2 / ct.vAx(x[ix]) ** 2 - ct.omega_n ** 2) \
		           * np.sum(1j * ct.k_perp * B_par1n * ct.I3 - \
		                    np.sin(ct.alpha) * B_par2n * ct.I5, axis = 0)
	u_perp2n[:,ix] = 1j * omega / (omega ** 2 / ct.vAx(x[ix]) ** 2 - ct.varpi_n ** 2) \
	               * np.sum(1j * ct.k_perp * B_par2n * ct.I4 - \
		                    np.sin(ct.alpha) * B_par1n * ct.I6, axis = 0)

ux = np.zeros((nz,nx), dtype = complex)
b_par = np.zeros((nz,nx), dtype = complex)
u_perp = np.zeros((nz,nx), dtype = complex)
for n in range(ct.N+1):
	Ux1n = np.tile(np.array([ux1n[n,:]]), (nz, 1))
	Ux2n = np.tile(np.array([ux2n[n,:]]), (nz, 1))
	B_par1n = np.tile(np.array([b_par1n[n,:]]), (nz, 1))
	B_par2n = np.tile(np.array([b_par2n[n,:]]), (nz, 1))
	U_perp1n = np.tile(np.array([u_perp1n[n,:]]), (nz, 1))
	U_perp2n = np.tile(np.array([u_perp2n[n,:]]), (nz, 1))
	ux += Ux1n * eg.phi(Z,n) + Ux2n * eg.varphi(Z,n)
	b_par += B_par1n * eg.phi(Z,n) + B_par2n * eg.varphi(Z,n)
	u_perp += U_perp1n * eg.phi(Z,n) + U_perp2n * eg.varphi(Z,n)

iz0 = 3 * nz // 4
ix0 = nx // 2
z0 = z[iz0]
x0 = x[ix0]

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(x, ux[iz0,:].real)
# ax.plot(x, ux_ana(x,z0).real)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(x, ux[iz0,:].imag)
# ax.plot(x, ux_ana(x,z0).imag)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(x, b_par[iz0,:].real)
# ax.plot(x, b_par_ana(Z)[iz0,:].real)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(x, b_par[iz0,:].imag)
# ax.plot(x, b_par_ana(Z)[iz0,:].imag)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(x, u_perp[iz0,:].real)
# ax.plot(x, u_perp_ana(x,z0).real)
# ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(x, u_perp[iz0,:].imag)
# ax.plot(x, u_perp_ana(x,z0).imag)
# ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(z, ux[:,ix0].real)
# ax.plot(z, ux_ana(x0,z).real)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(z, ux[:,ix0].imag)
# ax.plot(z, ux_ana(x0,z).imag)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(z, b_par[:,ix0].real)
# ax.plot(z, b_par_ana(z).real)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(z, b_par[:,ix0].imag)
# ax.plot(z, b_par_ana(z).imag)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(z, u_perp[:,ix0].real)
# ax.plot(z, u_perp_ana(x0,z).real)
# ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(z, u_perp[:,ix0].imag)
# ax.plot(z, u_perp_ana(x0,z).imag)
# ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Z, ux.real, cmap=cm.cool)
# ax.set_xlabel('x')
# ax.set_ylabel('z')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Z, ux.imag, cmap=cm.cool)
# ax.set_xlabel('x')
# ax.set_ylabel('z')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Z, b_par.real, cmap=cm.cool)
ax.set_xlabel('x')
ax.set_ylabel('z')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Z, b_par.imag, cmap=cm.cool)
ax.set_xlabel('x')
ax.set_ylabel('z')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Z, u_perp.real, cmap=cm.cool)
# ax.set_xlabel('x')
# ax.set_ylabel('z')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Z, u_perp.imag, cmap=cm.cool)
# ax.set_xlabel('x')
# ax.set_ylabel('z')

plt.show()