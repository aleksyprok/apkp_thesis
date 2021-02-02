import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def P(f):
	f = np.array(f)
	P = np.zeros_like(f)
	P += 10 ** -6    * f ** -1.34 * (10 ** -3.2 <= f) * (f <  10 ** -2.7)
	P += 10 ** -2.38              * (10 ** -2.7 <= f) * (f <  10 ** -2.4)
	P += 10 ** -6.05 * f ** -1.53 * (10 ** -2.4 <= f) * (f <= 10 ** -1.8)
	return P

fig = plt.figure()
fig_size = fig.get_size_inches()
fig_size[1] = fig_size[1] / 2
fig.set_size_inches(fig_size)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.2)

f_min = 10 ** -3.2
f_max = 10 ** -1.8
nf = 100
f = np.linspace(f_min, f_max, nf)
ax = fig.add_subplot(121)
ax.plot(f, P(f))
ax.text(0.925, 1.1, \
	r'$P(f)\ (km^2\,s^{-2})$', \
	fontsize = 12, \
	transform=ax.transAxes)
ax.set_xlabel(r'$f\ (Hz)$')


f_min = -3.2
f_max = -1.8
nf = 100
f = np.logspace(f_min, f_max, nf)
ax = fig.add_subplot(122)
ax.loglog(f, P(f))
ax.set_xlabel(r'$f\ (Hz)$')

fig.savefig('temp_figures/power_spectrum_morton.png', bbox_inches = 'tight', dpi = 150)

plt.show()