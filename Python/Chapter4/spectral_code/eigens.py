import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy import optimize
from scipy import integrate
import control as ct

def det_omega_n(omega_n):

	kznm = omega_n / ct.vAm / np.cos(ct.alpha)
	kznp = omega_n / ct.vAp / np.cos(ct.alpha)

	return kznp * np.sin(kznp * ct.lz) * np.cos(kznm * ct.lz) + \
		   kznm * np.sin(kznm * ct.lz) * np.cos(kznp * ct.lz)

def det_varpi_n(varpi_n):

	k_barnm = varpi_n / ct.vAm / np.cos(ct.alpha)
	k_barnp = varpi_n / ct.vAp / np.cos(ct.alpha)

	return k_barnp * np.sin(k_barnm * ct.lz) * np.cos(k_barnp * ct.lz) + \
		   k_barnm * np.sin(k_barnp * ct.lz) * np.cos(k_barnm * ct.lz)

def det_omega_n_prime(omega_n):

	kznm = omega_n / ct.vAm / np.cos(ct.alpha)
	kznp = omega_n / ct.vAp / np.cos(ct.alpha)

	return kznp ** 2   * ct.lz * np.cos(kznp * ct.lz) * np.cos(kznm * ct.lz) - \
		   kznp * kznm * ct.lz * np.sin(kznp * ct.lz) * np.sin(kznm * ct.lz) + \
		   kznm ** 2   * ct.lz * np.cos(kznm * ct.lz) * np.cos(kznp * ct.lz) - \
		   kznm * kznp * ct.lz * np.sin(kznm * ct.lz) * np.sin(kznp * ct.lz)

def det_varpi_n_prime(varpi_n):

	k_barnm = varpi_n / ct.vAm / np.cos(ct.alpha)
	k_barnp = varpi_n / ct.vAp / np.cos(ct.alpha)

	return k_barnp * k_barnm * ct.lz * np.cos(k_barnm * ct.lz) * np.cos(k_barnp * ct.lz) - \
		   k_barnp ** 2      * ct.lz * np.sin(k_barnm * ct.lz) * np.sin(k_barnp * ct.lz) + \
		   k_barnm * k_barnp * ct.lz * np.cos(k_barnp * ct.lz) * np.cos(k_barnm * ct.lz) - \
		   k_barnm ** 2      * ct.lz * np.sin(k_barnp * ct.lz) * np.sin(k_barnm * ct.lz)

def phi(z, n):
	return ct.A[n] * ct.a[n] * np.cos(ct.kzm[n] * (z + ct.lz)) * (z <  0) + \
		   ct.A[n] * ct.c[n] * np.cos(ct.kzp[n] * (z - ct.lz)) * (z >= 0)

def varphi(z, n):
	return ct.B[n] * ct.b[n] * np.sin(ct.k_barm[n] * (z + ct.lz)) * (z <  0) + \
		   ct.B[n] * ct.d[n] * np.sin(ct.k_barp[n] * (z - ct.lz)) * (z >= 0)

def phi_prime(z, n):
	return -ct.kzm[n] * ct.A[n] * ct.a[n] * np.sin(ct.kzm[n] * (z + ct.lz)) * (z <  0) - \
		    ct.kzp[n] * ct.A[n] * ct.c[n] * np.sin(ct.kzp[n] * (z - ct.lz)) * (z >= 0)

def phi_prime2(z, n):
	return -ct.kzm[n] ** 2 * ct.A[n] * ct.a[n] * np.cos(ct.kzm[n] * (z + ct.lz)) * (z <  0) - \
		    ct.kzp[n] ** 2 * ct.A[n] * ct.c[n] * np.cos(ct.kzp[n] * (z - ct.lz)) * (z >= 0)

def varphi_prime(z, n):
	return ct.k_barm[n] * ct.B[n] * ct.b[n] * np.cos(ct.k_barm[n] * (z + ct.lz)) * (z <  0) + \
		   ct.k_barp[n] * ct.B[n] * ct.d[n] * np.cos(ct.k_barp[n] * (z - ct.lz)) * (z >= 0)

def calc_eigenfrequencies():

	omega_array = np.linspace(ct.omega_min_guess / 2, 2 * ct.omega_max_guess, ct.n_omega)
	omega_ana_1 = np.pi * ct.vAp * np.cos(ct.alpha) / ct.Lz

	### Calculate omega_n ###

	# Calculate initial guesses
	cs = CubicSpline(omega_array, det_omega_n(omega_array))
	omega_n_guess = CubicSpline.roots(cs)

	for i in range(1,ct.N+1):
		ct.omega_n[i] = omega_n_guess[i-1]

	# If omega_n is close to an integer multiple of omega_ana_1 then
	# assign it to that value.
	n1 = ct.omega_n / omega_ana_1
	logic1 = (np.abs(n1 - np.round(n1)) < 1e-6) * 1
	ct.omega_n += (np.round(n1) * omega_ana_1 - ct.omega_n) * logic1
	ct.Omega_n = np.tile(np.array([ct.omega_n]).T, (1, ct.N+1))

	### Calculate varpi_n ###

	# Calculate initial guesses
	cs = CubicSpline(omega_array, det_varpi_n(omega_array))
	varpi_n_guess = CubicSpline.roots(cs)

	for i in range(1,ct.N+1):
		ct.varpi_n[i] = varpi_n_guess[i-1]

	n1 = ct.varpi_n / omega_ana_1
	logic1 = (np.abs(n1 - np.round(n1)) < 1e-6) * 1
	ct.varpi_n += (np.round(n1) * omega_ana_1 - ct.varpi_n) * logic1
	ct.Varpi_n = np.tile(np.array([ct.varpi_n]).T, (1, ct.N+1))

def get_parameters():

	ct.kzm = ct.omega_n / ct.vAm / np.cos(ct.alpha)
	ct.kzp = ct.omega_n / ct.vAp / np.cos(ct.alpha)
	ct.k_barm = ct.varpi_n / ct.vAm / np.cos(ct.alpha)
	ct.k_barp = ct.varpi_n / ct.vAp / np.cos(ct.alpha)

	ct.a = np.cos(ct.kzp * ct.lz)          * (np.abs(np.cos(ct.kzp * ct.lz)) >= 1e-6) \
	     + ct.kzp * np.sin(ct.kzp * ct.lz) * (np.abs(np.cos(ct.kzp * ct.lz)) <  1e-6)

	ct.c = np.cos(ct.kzm * ct.lz)          * (np.abs(np.cos(ct.kzp * ct.lz)) >= 1e-6) \
	     - ct.kzm * np.sin(ct.kzm * ct.lz) * (np.abs(np.cos(ct.kzp * ct.lz)) <  1e-6)

	ct.b =  np.sin(ct.k_barp * ct.lz)             * (np.abs(np.sin(ct.k_barp * ct.lz)) >= 1e-6) \
	     + ct.k_barp * np.cos(ct.k_barp * ct.lz)  * (np.abs(np.sin(ct.k_barp * ct.lz)) <  1e-6)

	ct.d = -np.sin(ct.k_barm * ct.lz)             * (np.abs(np.sin(ct.k_barp * ct.lz)) >= 1e-6) \
	     + ct.k_barm * np.cos(ct.k_barm * ct.lz)  * (np.abs(np.sin(ct.k_barp * ct.lz)) <  1e-6)

	ct.Kzm = np.tile(np.array([ct.kzm]).T, (1, ct.N+1))
	ct.Kzp = np.tile(np.array([ct.kzp]).T, (1, ct.N+1))
	ct.K_barm = np.tile(np.array([ct.k_barm]).T, (1, ct.N+1))
	ct.K_barp = np.tile(np.array([ct.k_barp]).T, (1, ct.N+1))
	ct.aa = np.tile(np.array([ct.a]).T, (1, ct.N+1))
	ct.bb = np.tile(np.array([ct.b]).T, (1, ct.N+1))
	ct.cc = np.tile(np.array([ct.c]).T, (1, ct.N+1))
	ct.dd = np.tile(np.array([ct.d]).T, (1, ct.N+1))

def calc_inner_products():

	dummy = np.zeros(ct.N+1)
	dummy[0] = 1

	# Calc A
	w1 = ct.a ** 2 / (2 * ct.vAm ** 2 * ct.kzm + dummy)
	w2 = ct.c ** 2 / (2 * ct.vAp ** 2 * ct.kzp + dummy)
	w3 = w1 * (2 * ct.kzm * ct.lz + np.sin(2 * ct.kzm * ct.lz)) \
	   + w2 * (2 * ct.kzp * ct.lz + np.sin(2 * ct.kzp * ct.lz))
	w3[0] = 2 * ct.lz * (1 / ct.vAm ** 2 + 1 / ct.vAp ** 2)
	ct.A = 1 / np.sqrt(w3)
	ct.AA = np.tile(np.array([ct.A]).T, (1, ct.N+1))

	# Calc B
	w1 = ct.b ** 2 / (2 * ct.vAm ** 2 * ct.k_barm + dummy)
	w2 = ct.d ** 2 / (2 * ct.vAp ** 2 * ct.k_barp + dummy)
	w3 = w1 * (2 * ct.k_barm * ct.lz - np.sin(2 * ct.k_barm * ct.lz)) \
	   + w2 * (2 * ct.k_barp * ct.lz - np.sin(2 * ct.k_barp * ct.lz))
	ct.B[1:] = 1 / np.sqrt(w3[1:])
	ct.B[0] = 0
	ct.BB = np.tile(np.array([ct.B]).T, (1, ct.N+1))

	# Calc I1
	w1 = 2 * ct.aa * ct.aa.T / ct.vAm ** 4 \
	   * ct.Kzm   * np.sin(ct.Kzm   * ct.lz) * np.cos(ct.Kzm.T * ct.lz)
	w2 = 2 * ct.aa * ct.aa.T / ct.vAm ** 4 \
	   * ct.Kzm.T * np.sin(ct.Kzm.T * ct.lz) * np.cos(ct.Kzm   * ct.lz)
	w3 = 2 * ct.cc * ct.cc.T / ct.vAp ** 4 \
	   * ct.Kzp   * np.sin(ct.Kzp   * ct.lz) * np.cos(ct.Kzp.T * ct.lz)
	w4 = 2 * ct.cc * ct.cc.T / ct.vAp ** 4 \
	   * ct.Kzp.T * np.sin(ct.Kzp.T * ct.lz) * np.cos(ct.Kzp   * ct.lz)
	w5 = (w1 - w2) / (ct.Kzm ** 2 - ct.Kzm.T ** 2 + np.identity(ct.N+1)) \
	   + (w3 - w4) / (ct.Kzp ** 2 - ct.Kzp.T ** 2 + np.identity(ct.N+1))
	ct.I1 = ct.AA * ct.AA.T * w5 * (np.ones((ct.N+1,ct.N+1)) - np.identity(ct.N+1))
	w1 = ct.a ** 2 / (2 * ct.vAm ** 4 * ct.kzm + dummy) \
	   * (2 * ct.kzm * ct.lz + np.sin(2 * ct.kzm * ct.lz))
	w2 = ct.c ** 2 / (2 * ct.vAp ** 4 * ct.kzp + dummy) \
	   * (2 * ct.kzp * ct.lz + np.sin(2 * ct.kzp * ct.lz))
	w3 = ct.A ** 2 * (w1 + w2)
	w3[0] = ct.A[0] ** 2 * 2 * ct.lz * (1 / ct.vAm ** 4 + 1 / ct.vAp ** 4)
	ct.I1 += np.diag(w3)

	# Calc I2
	w1 = 2 * ct.bb * ct.bb.T / ct.vAm ** 4 \
	   * ct.K_barm.T * np.sin(ct.K_barm   * ct.lz) * np.cos(ct.K_barm.T * ct.lz)
	w2 = 2 * ct.bb * ct.bb.T / ct.vAm ** 4 \
	   * ct.K_barm   * np.sin(ct.K_barm.T * ct.lz) * np.cos(ct.K_barm   * ct.lz)
	w3 = 2 * ct.dd * ct.dd.T / ct.vAp ** 4 \
	   * ct.K_barp.T * np.sin(ct.K_barp   * ct.lz) * np.cos(ct.K_barp.T * ct.lz)
	w4 = 2 * ct.dd * ct.dd.T / ct.vAp ** 4 \
	   * ct.K_barp   * np.sin(ct.K_barp.T * ct.lz) * np.cos(ct.K_barp   * ct.lz)
	w5 = (w1 - w2) / (ct.K_barm ** 2 - ct.K_barm.T ** 2 + np.identity(ct.N+1)) \
	   + (w3 - w4) / (ct.K_barp ** 2 - ct.K_barp.T ** 2 + np.identity(ct.N+1))
	ct.I2 = ct.BB * ct.BB.T * w5 * (np.ones((ct.N+1,ct.N+1)) - np.identity(ct.N+1))
	w1 = ct.b ** 2 / (2 * ct.vAm ** 4 * ct.k_barm + dummy) \
	   * (2 * ct.k_barm * ct.lz - np.sin(2 * ct.k_barm * ct.lz))
	w2 = ct.d ** 2 / (2 * ct.vAp ** 4 * ct.k_barp + dummy) \
	   * (2 * ct.k_barp * ct.lz - np.sin(2 * ct.k_barp * ct.lz))
	w3 = ct.B ** 2 * (w1 + w2)
	ct.I2 += np.diag(w3)

	# Calc I3
	w1 = 2 * ct.aa * ct.aa.T \
	   * ct.Kzm   * np.sin(ct.Kzm   * ct.lz) * np.cos(ct.Kzm.T * ct.lz)
	w2 = 2 * ct.aa * ct.aa.T \
	   * ct.Kzm.T * np.sin(ct.Kzm.T * ct.lz) * np.cos(ct.Kzm   * ct.lz)
	w3 = 2 * ct.cc * ct.cc.T \
	   * ct.Kzp   * np.sin(ct.Kzp   * ct.lz) * np.cos(ct.Kzp.T * ct.lz)
	w4 = 2 * ct.cc * ct.cc.T \
	   * ct.Kzp.T * np.sin(ct.Kzp.T * ct.lz) * np.cos(ct.Kzp   * ct.lz)
	w5 = (w1 - w2) / (ct.Kzm ** 2 - ct.Kzm.T ** 2 + np.identity(ct.N+1)) \
	   + (w3 - w4) / (ct.Kzp ** 2 - ct.Kzp.T ** 2 + np.identity(ct.N+1))
	ct.I3 = ct.AA * ct.AA.T * w5 * (np.ones((ct.N+1,ct.N+1)) - np.identity(ct.N+1))
	w1 = ct.a ** 2 / (2 * ct.kzm + dummy) \
	   * (2 * ct.kzm * ct.lz + np.sin(2 * ct.kzm * ct.lz))
	w2 = ct.c ** 2 / (2 * ct.kzp + dummy) \
	   * (2 * ct.kzp * ct.lz + np.sin(2 * ct.kzp * ct.lz))
	w3 = ct.A ** 2 * (w1 + w2)
	w3[0] = ct.A[0] ** 2 * 4 * ct.lz
	ct.I3 += np.diag(w3)

	# Calc I4
	w1 = 2 * ct.bb * ct.bb.T \
	   * ct.K_barm.T * np.sin(ct.K_barm   * ct.lz) * np.cos(ct.K_barm.T * ct.lz)
	w2 = 2 * ct.bb * ct.bb.T \
	   * ct.K_barm   * np.sin(ct.K_barm.T * ct.lz) * np.cos(ct.K_barm   * ct.lz)
	w3 = 2 * ct.dd * ct.dd.T \
	   * ct.K_barp.T * np.sin(ct.K_barp   * ct.lz) * np.cos(ct.K_barp.T * ct.lz)
	w4 = 2 * ct.dd * ct.dd.T \
	   * ct.K_barp   * np.sin(ct.K_barp.T * ct.lz) * np.cos(ct.K_barp   * ct.lz)
	w5 = (w1 - w2) / (ct.K_barm ** 2 - ct.K_barm.T ** 2 + np.identity(ct.N+1)) \
	   + (w3 - w4) / (ct.K_barp ** 2 - ct.K_barp.T ** 2 + np.identity(ct.N+1))
	ct.I4 = ct.BB * ct.BB.T * w5 * (np.ones((ct.N+1,ct.N+1)) - np.identity(ct.N+1))
	w1 = ct.b ** 2 / (2 * ct.k_barm + dummy) \
	   * (2 * ct.k_barm * ct.lz - np.sin(2 * ct.k_barm * ct.lz))
	w2 = ct.d ** 2 / (2 * ct.k_barp + dummy) \
	   * (2 * ct.k_barp * ct.lz - np.sin(2 * ct.k_barp * ct.lz))
	w3 = ct.B ** 2 * (w1 + w2)
	ct.I4 += np.diag(w3)

	# Calc I5
	omega_varpi_eq = (ct.omega_n == ct.varpi_n) * 1
	w1 = 2 * ct.bb * ct.aa.T * ct.K_barm \
	   * ct.K_barm * np.sin(ct.K_barm * ct.lz) * np.cos(ct.Kzm.T  * ct.lz)
	w2 = 2 * ct.bb * ct.aa.T * ct.K_barm \
	   * ct.Kzm.T  * np.sin(ct.Kzm.T  * ct.lz) * np.cos(ct.K_barm * ct.lz)
	w3 = 2 * ct.dd * ct.cc.T * ct.K_barp \
	   * ct.K_barp * np.sin(ct.K_barp * ct.lz) * np.cos(ct.Kzp.T  * ct.lz)
	w4 = 2 * ct.dd * ct.cc.T * ct.K_barp \
	   * ct.Kzp.T  * np.sin(ct.Kzp.T  * ct.lz) * np.cos(ct.K_barp * ct.lz)
	w5 = (w1 - w2) / (ct.K_barm ** 2 - ct.Kzm.T ** 2 + np.diag(omega_varpi_eq)) \
	   + (w3 - w4) / (ct.K_barp ** 2 - ct.Kzp.T ** 2 + np.diag(omega_varpi_eq))
	ct.I5 = ct.BB * ct.AA.T * w5 * (np.ones((ct.N+1,ct.N+1)) - np.diag(omega_varpi_eq))
	w1 = ct.b * ct.a / 2 \
	   * (2 * ct.kzm * ct.lz + np.sin(2 * ct.kzm * ct.lz))
	w2 = ct.d * ct.c / 2 \
	   * (2 * ct.kzp * ct.lz + np.sin(2 * ct.kzp * ct.lz))
	w3 = ct.A * ct.B * (w1 + w2) * omega_varpi_eq
	ct.I5 += np.diag(w3)

	# Calc I6
	omega_varpi_eq = (ct.omega_n == ct.varpi_n) * 1
	w1 = 2 * ct.aa * ct.bb.T * ct.Kzm \
	   * ct.K_barm.T * np.sin(ct.Kzm      * ct.lz) * np.cos(ct.K_barm.T * ct.lz)
	w2 = 2 * ct.aa * ct.bb.T * ct.Kzm \
	   * ct.Kzm      * np.sin(ct.K_barm.T * ct.lz) * np.cos(ct.Kzm      * ct.lz)
	w3 = 2 * ct.cc * ct.dd.T * ct.Kzp \
	   * ct.K_barp.T * np.sin(ct.Kzp     * ct.lz) * np.cos(ct.K_barp.T * ct.lz)
	w4 = 2 * ct.cc * ct.dd.T * ct.Kzp \
	   * ct.Kzp     * np.sin(ct.K_barp.T * ct.lz) * np.cos(ct.Kzp    * ct.lz)
	w5 = -(w1 - w2) / (ct.Kzm ** 2 - ct.K_barm.T ** 2 + np.diag(omega_varpi_eq)) \
	   -  (w3 - w4) / (ct.Kzp ** 2 - ct.K_barp.T ** 2 + np.diag(omega_varpi_eq))
	ct.I6 = ct.AA * ct.BB.T * w5 * (np.ones((ct.N+1,ct.N+1)) - np.diag(omega_varpi_eq))
	w1 = ct.a * ct.b / 2 \
	   * (2 * ct.kzm * ct.lz - np.sin(2 * ct.kzm * ct.lz))
	w2 = ct.c * ct.d / 2 \
	   * (2 * ct.kzp * ct.lz - np.sin(2 * ct.kzp * ct.lz))
	w3 = -ct.A * ct.B * (w1 + w2) * omega_varpi_eq
	ct.I6 += np.diag(w3)

	# Calc I7
	omega_varpi_eq = (ct.omega_n == ct.varpi_n) * 1
	w1 = 2 * ct.aa * ct.bb.T * ct.Kzm / ct.vAm ** 2 \
	   * ct.K_barm.T * np.sin(ct.Kzm      * ct.lz) * np.cos(ct.K_barm.T * ct.lz)
	w2 = 2 * ct.aa * ct.bb.T * ct.Kzm / ct.vAm ** 2 \
	   * ct.Kzm      * np.sin(ct.K_barm.T * ct.lz) * np.cos(ct.Kzm      * ct.lz)
	w3 = 2 * ct.cc * ct.dd.T * ct.Kzp / ct.vAp ** 2 \
	   * ct.K_barp.T * np.sin(ct.Kzp     * ct.lz) * np.cos(ct.K_barp.T * ct.lz)
	w4 = 2 * ct.cc * ct.dd.T * ct.Kzp / ct.vAp ** 2 \
	   * ct.Kzp     * np.sin(ct.K_barp.T * ct.lz) * np.cos(ct.Kzp    * ct.lz)
	w5 = -(w1 - w2) / (ct.Kzm ** 2 - ct.K_barm.T ** 2 + np.diag(omega_varpi_eq)) \
	   -  (w3 - w4) / (ct.Kzp ** 2 - ct.K_barp.T ** 2 + np.diag(omega_varpi_eq))
	ct.I7 = ct.AA * ct.BB.T * w5 * (np.ones((ct.N+1,ct.N+1)) - np.diag(omega_varpi_eq))
	w1 = ct.a * ct.b / (2 * ct.vAm ** 2) \
	   * (2 * ct.kzm * ct.lz - np.sin(2 * ct.kzm * ct.lz))
	w2 = ct.c * ct.d / (2 * ct.vAp ** 2) \
	   * (2 * ct.kzp * ct.lz - np.sin(2 * ct.kzp * ct.lz))
	w3 = -ct.A * ct.B * (w1 + w2) * omega_varpi_eq
	ct.I7 += np.diag(w3)