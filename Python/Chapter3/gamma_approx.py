import numpy as np
from scipy.special import zeta

N = 100
alpha = 1
omega = np.pi * 1e-2
eta = 1e1
a = 1e6

gamma = omega * (4 * eta / (3 * omega * a ** 2)) ** (1/3) \
	  * (3 * N ** (2 / 3) / 2 + zeta(1/3)) / (np.log(N) + np.euler_gamma)

print("{:1.2E}".format(gamma))

mp = 1.6726219e-27
n = 1e15
rho = 1e-12
n_new = rho / mp
T = 1e6
B = 1e-3
omega_tau = 0.82e5 * n_new / n
coulomb_log = 20
eta0 = 2.21e-16 * T ** (5 / 2) / coulomb_log
eta1 = 0.3 * eta0 / omega_tau ** 2
nu = eta1 / rho

print("{:1.2E}".format(nu))
print("{:1.2E}".format(n_new))
print("{:1.2E}".format(omega_tau))
