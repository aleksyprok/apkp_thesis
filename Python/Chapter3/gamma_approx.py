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
