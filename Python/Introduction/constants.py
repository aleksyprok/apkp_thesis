import numpy as np

def omega_ce_gauss(B):
	return e_gauss * B / me_gauss / c_gauss

def omega_ce_SI(B):
	return e_SI * B / me_SI

def omega_ci_gauss(B):
	return e_gauss * B / mp_gauss / c_gauss

def omega_ci_SI(B):
	return e_SI * B / mp_SI

def nu_e_gauss(n, T):
	return 2.91e-6 * n * coulomb_log / T ** (3 / 2)

def nu_e_SI(n, T):
	return 3.64e-6 * n * coulomb_log / T ** (3 / 2)

def nu_i_gauss(n, T):
	return 4.80e-8 * n * coulomb_log / T ** (3 / 2)

def nu_i_SI(n, T):
	return 6.00e-8 * n * coulomb_log / T ** (3 / 2)

def v_Te_gauss(T):
	return np.sqrt(kB_gauss * T / me_gauss)

def v_Te_SI(T):
	return np.sqrt(kB_SI * T / me_SI)

def v_Ti_gauss(T):
	return np.sqrt(kB_gauss * T / mp_gauss)

def v_Ti_SI(T):
	return np.sqrt(kB_SI * T / mp_SI)

def r_e_gauss(B, T):
	return v_Te_gauss(T) / omega_ce_gauss(B)

def r_e_SI(B, T):
	return v_Te_SI(T) / omega_ce_SI(B)

def r_i_gauss(B, T):
	return v_Ti_gauss(T) / omega_ci_gauss(B)

def r_i_SI(B, T):
	return v_Ti_SI(T) / omega_ci_SI(B)

def lambda_e_gauss(n, T):
	return v_Te_gauss(T) / nu_e_gauss(n,T)

def lambda_e_SI(n, T):
	return v_Te_SI(T) / nu_e_SI(n,T)

def lambda_i_gauss(n, T):
	return v_Ti_gauss(T) / nu_i_gauss(n,T)

def lambda_i_SI(n, T):
	return v_Ti_SI(T) / nu_i_SI(n,T)

def omega_ce_nu_e1(B,n,T):
	return omega_ce_SI(B) / nu_e_SI(n,T)

def omega_ce_nu_e2(B,n,T):
	return 1 / 3.64e-6 * e_SI * B * T ** (3 / 2) / (me_SI * n * coulomb_log)

def omega_ci_nu_i1(B,n,T):
	return omega_ci_SI(B) / nu_i_SI(n,T)

def omega_ci_nu_i2(B,n,T):
	return 1 / 6.00e-8 * e_SI * B * T ** (3 / 2) / (mp_SI * n * coulomb_log)

def lambda_e(n, T):
	return 1 / (3.64e-6) * np.sqrt(kB_SI / me_SI) * T ** 2 / n / coulomb_log

def lambda_i(n, T):
	return 1 / (6.00e-8) * np.sqrt(kB_SI / mp_SI) * T ** 2 / n / coulomb_log

def r_e(B, T):
	return np.sqrt(me_SI * kB_SI * T) / (e_SI * B)

e_gauss = 4.8032e-10
me_gauss = 9.10938356e-28
mp_gauss = 1.6726219e-24
c_gauss = 2.9979e10
kB_gauss = 1.60e-12

e_SI = 1.60217662e-19
me_SI = 9.10938356e-31
mp_SI = 1.6726219e-27
c_SI = 2.9979e8
kB_SI = 1.3807e-23

coulomb_log = 20
B = 1e-3
T = 1e6
n = 1e15
print('{:e}'.format(omega_ce_nu_e1(B,n,T)))
print('{:e}'.format(omega_ce_nu_e2(B,n,T)))
print('{:e}'.format(lambda_e(n,T)))
print('{:e}'.format(lambda_i(n,T)))
print('{:e}'.format(r_e(B,T)))
print('{:e}'.format(omega_ci_nu_i1(B,n,T)))
print('{:e}'.format(omega_ci_nu_i2(B,n,T)))