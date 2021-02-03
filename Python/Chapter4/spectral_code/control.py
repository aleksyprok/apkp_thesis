import numpy as np
import matplotlib.pyplot as plt

def vAx(x):
	return 1 + x / a0

def vA(x,z):
	return vAx(x) * (vAm * (z <  0) + \
		             vAp * (z >= 0))

vAm = 1 / 21
vAp = 1
Lz = 1
lz = Lz / 2
alpha = 0.25 * np.pi
k_perp = 0.5 * np.pi * np.sin(alpha) / Lz
a0 = 1

errtol = 1e-10

N = 128
omega_min_guess = np.pi * vAm * np.cos(alpha) / Lz
omega_max_guess = N * omega_min_guess
n_omega = 1024 * N
omega_n = np.zeros(N+1)
varpi_n = np.zeros(N+1)

kzm = np.zeros(N+1)
kzp = np.zeros(N+1)
k_barm = np.zeros(N+1)
k_barp = np.zeros(N+1)
a = np.zeros(N+1)
b = np.zeros(N+1)
c = np.zeros(N+1)
d = np.zeros(N+1)
A = np.zeros(N+1)
B = np.zeros(N+1)

Kzm = np.zeros((N+1,N+1))
Kzp = np.zeros((N+1,N+1))
K_barm = np.zeros((N+1,N+1))
K_barp = np.zeros((N+1,N+1))
aa = np.zeros((N+1,N+1))
bb = np.zeros((N+1,N+1))
cc = np.zeros((N+1,N+1))
dd = np.zeros((N+1,N+1))

AA = np.zeros((N+1,N+1))
BB = np.zeros((N+1,N+1))
I1 = np.zeros((N+1,N+1))
I2 = np.zeros((N+1,N+1))
I3 = np.zeros((N+1,N+1))
I4 = np.zeros((N+1,N+1))
I5 = np.zeros((N+1,N+1))
I6 = np.zeros((N+1,N+1))
I7 = np.zeros((N+1,N+1))