import numpy as np

omega = np.pi * 1e-2
eta = 10
a = 1e6
vAp = 1e6
vAm = 1e3

qty = (omega ** 2 * eta * a / 6) ** (2 / 3) / (vAp * vAm)
lx = 2 * np.pi * (eta * a / (6 * omega)) ** (1 / 3)

kx = (6 * omega / (eta * a)) ** (1 / 3)
kzp = omega / vAp
kzm = omega / vAm

lzp = 2 * np.pi / kzp
lzm = 2 * np.pi / kzm

print('{:e}'.format(qty))
print('{:e}'.format(lx))
print('{:e}'.format(lzp))
print('{:e}'.format(lzm))