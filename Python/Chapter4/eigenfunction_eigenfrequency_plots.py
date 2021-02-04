import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib.lines import Line2D   

def det_omega_n(sigma, omega_n):

	vAm = vAp / sigma

	kznm = omega_n / vAm / np.cos(alpha)
	kznp = omega_n / vAp / np.cos(alpha)

	return kznp * np.sin(kznp * lz) * np.cos(kznm * lz) + \
		   kznm * np.sin(kznm * lz) * np.cos(kznp * lz)

def det_varpi_n(sigma, varpi_n):

	vAm = vAp / sigma

	kbar_nm = varpi_n / vAm / np.cos(alpha)
	kbar_np = varpi_n / vAp / np.cos(alpha)

	return kbar_np * np.sin(kbar_nm * lz) * np.cos(kbar_np * lz) + \
		   kbar_nm * np.sin(kbar_np * lz) * np.cos(kbar_nm * lz)

def phi(z, n):
	return A[n] * a[n] * np.cos(kzm[n] * (z + lz)) * (z <  0) + \
		   A[n] * c[n] * np.cos(kzp[n] * (z - lz)) * (z >= 0)

def varphi(z, n):
	return B[n] * b[n] * np.sin(k_barm[n] * (z + lz)) * (z <  0) + \
		   B[n] * d[n] * np.sin(k_barp[n] * (z - lz)) * (z >= 0)

def phi_prime(z, n):
	return -kzm[n] * A[n] * a[n] * np.sin(kzm[n] * (z + lz)) * (z <  0) - \
		    kzp[n] * A[n] * c[n] * np.sin(kzp[n] * (z - lz)) * (z >= 0)


def varphi_prime(z, n):
	return k_barm[n] * B[n] * b[n] * np.cos(k_barm[n] * (z + lz)) * (z <  0) + \
		   k_barp[n] * B[n] * d[n] * np.cos(k_barp[n] * (z - lz)) * (z >= 0)


vAp = 1
sigma = 5
vAm = vAp / sigma
Lz = 1
lz = Lz / 2
alpha = 0.25 * np.pi

nz = 1024
z_min = -Lz
z_max =  Lz
dz = (z_max - z_min) / nz
z = np.linspace(z_min, z_max, nz)

omega_norm = np.pi * vAp * np.cos(alpha) / Lz

n_harmonics = 20
omega_min_guess = np.pi * vAm * np.cos(alpha) / Lz
# omega_max = n_harmonics * omega_min_guess
omega_max = 2 * omega_norm
n_omega = 64 * n_harmonics
omega_array = np.linspace(0, omega_max, n_omega)

n_omega_big = 1024 * n_harmonics
omega_array_big = np.linspace(omega_min_guess / 2, 2 * n_harmonics * omega_min_guess, n_omega)

sigma_min = 1
sigma_max = 5
n_sigma = 128
sigma_array = np.linspace(sigma_min, sigma_max, n_sigma)

Sigma, Omega = np.meshgrid(sigma_array, omega_array)

omega_n = np.zeros(n_harmonics+1)
cs1 = CubicSpline(omega_array_big, det_omega_n(sigma, omega_array_big))
omega_n[1:n_harmonics+1] = CubicSpline.roots(cs1)[0:n_harmonics]

varpi_n = np.zeros(n_harmonics+1)
cs2 = CubicSpline(omega_array_big, det_varpi_n(sigma, omega_array_big))
varpi_n[1:n_harmonics+1] = CubicSpline.roots(cs2)[0:n_harmonics]

kzm = omega_n / vAm / np.cos(alpha)
kzp = omega_n / vAp / np.cos(alpha)
k_barm = varpi_n / vAm / np.cos(alpha)
k_barp = varpi_n / vAp / np.cos(alpha)

a = np.cos(kzp * lz)          * (np.abs(np.cos(kzm * lz)) >= 1e-6) \
     + kzp * np.sin(kzp * lz) * (np.abs(np.cos(kzm * lz)) <  1e-6)

c = np.cos(kzm * lz)          * (np.abs(np.cos(kzm * lz)) >= 1e-6) \
     - kzm * np.sin(kzm * lz) * (np.abs(np.cos(kzm * lz)) <  1e-6)

b =  np.sin(k_barp * lz)             * (np.abs(np.sin(k_barm * lz)) >= 1e-6) \
     + k_barp * np.cos(k_barp * lz)  * (np.abs(np.sin(k_barm * lz)) <  1e-6)

d = -np.sin(k_barm * lz)             * (np.abs(np.sin(k_barm * lz)) >= 1e-6) \
     + k_barm * np.cos(k_barm * lz)  * (np.abs(np.sin(k_barm * lz)) <  1e-6)

dummy = np.zeros(n_harmonics+1)
dummy[0] = 1

# Calc A
w1 = a ** 2 / (2 * vAm ** 2 * kzm + dummy)
w2 = c ** 2 / (2 * vAp ** 2 * kzp + dummy)
w3 = w1 * (2 * kzm * lz + np.sin(2 * kzm * lz)) \
   + w2 * (2 * kzp * lz + np.sin(2 * kzp * lz))
w3[0] = 2 * lz * (1 / vAm ** 4 + 1 / vAp ** 2)
A = 1 / np.sqrt(w3)

# Calc B
w1 = b ** 2 / (2 * vAm ** 2 * k_barm + dummy)
w2 = d ** 2 / (2 * vAp ** 2 * k_barp + dummy)
w3 = w1 * (2 * k_barm * lz - np.sin(2 * k_barm * lz)) \
   + w2 * (2 * k_barp * lz - np.sin(2 * k_barp * lz))
B = 1 / np.sqrt(w3 + dummy)
B[0] = 0

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = 1.75 * fig_size[1]
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.45)

n = 1

ax = fig.add_subplot(421)
ax.plot(z, phi(z,n))
ax.set_title(r'$\phi_1(z)$')

ax = fig.add_subplot(422)
ax.plot(z, varphi(z,n), 'tab:orange')
ax.set_title(r'$\varphi_1(z)$')

n = 2

ax = fig.add_subplot(423)
ax.plot(z, phi(z,n))
ax.set_title(r'$\phi_2(z)$')

ax = fig.add_subplot(424)
ax.plot(z, varphi(z,n), 'tab:orange')
ax.set_title(r'$\varphi_2(z)$')

n = 3

ax = fig.add_subplot(425)
ax.plot(z, phi(z,n))
ax.set_title(r'$\phi_3(z)$')
ax.set_xlabel(r'$z / L_z$')

ax = fig.add_subplot(426)
ax.plot(z, varphi(z,n), 'tab:orange')
ax.set_title(r'$\varphi_3(z)$')
ax.set_xlabel(r'$z / L_z$')

ax = fig.add_subplot(427)
cp_levels = np.array([0])
ax.contour(Sigma, Omega / omega_norm, det_omega_n(Sigma, Omega), \
			levels = cp_levels, \
			colors = 'tab:blue', \
			linestyles = 'dashed')
ax.contour(Sigma, Omega / omega_norm, det_varpi_n(Sigma, Omega), \
			levels = cp_levels, \
			colors = 'tab:orange', \
			linestyles = 'dashed')
ax.set_xlabel(r'$v_{A+}\, \slash\, v_{A-}$')

line1 = Line2D([1],[1], color = "tab:blue", linestyle = 'dashed', label = r'$\omega_n\ \slash\ (\pi v_{A+} \cos(\alpha) / L_z)$')
line2 = Line2D([1],[1], color = "tab:orange", linestyle = 'dashed', label = r'$\varpi_n\ \slash\ (\pi v_{A+} \cos(\alpha) / L_z)$')
ax.add_line(line1)
ax.add_line(line2)
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.8,1.0))

ax.text(1.6, 0.3, \
	r"$v_{A-} = v_{A+}\ /\ $" + "{:2.1f}".format(sigma), \
	transform=ax.transAxes)

ax.annotate(r'$n=0$', xy = (5, 0), xytext = (5.5, 0),
            arrowprops = dict(arrowstyle="->"))
ax.annotate(r'$n=1$', xy = (5, 0.3), xytext = (5.5, 0.3),
            arrowprops = dict(arrowstyle="->"))
ax.annotate(r'$n=2$', xy = (5, 0.7), xytext = (5.5, 0.7),
            arrowprops = dict(arrowstyle="->"))
ax.annotate(r'$n=3$', xy = (5, 1), xytext = (5.5, 1),
            arrowprops = dict(arrowstyle="->"))
ax.annotate(r'$n=4$', xy = (5, 1.3), xytext = (5.5, 1.3),
            arrowprops = dict(arrowstyle="->"))
ax.annotate(r'$n=5$', xy = (5, 1.7), xytext = (5.5, 1.7),
            arrowprops = dict(arrowstyle="->"))

fig.savefig('temp_figures/eigenfunctions_and_eigenfrequencies.pdf', bbox_inches = 'tight')

plt.show(block = False)