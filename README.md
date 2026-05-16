import numpy as np
import matplotlib.pyplot as plt

Tc = 1.0
A = 1.0
t = np.linspace(0, 0.99*Tc, 500)
S = A / np.sqrt(Tc - t)
dS = A / (2 * (Tc - t)**1.5)
ell = 1.0
H = ell * dS / (S**2 + 1e-12)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,8))
ax1.plot(t, S, 'b-', linewidth=2.5)
ax1.set_xlabel('Time t', fontsize=12)
ax1.set_ylabel(r'$S(t)$', fontsize=12)
ax1.set_title('Sobolev-Stepanov norm evolution', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.plot(t, H, 'r-', linewidth=2.5, label=r'$H(t)$')
ax2.axhline(y=5, color='gold', linestyle='--', linewidth=2, label='Threshold')
ax2.fill_between(t, 0, H, where=(H>=5), color='red', alpha=0.2, label='Early warning zone')
ax2.set_xlabel('Time t', fontsize=12)
ax2.set_ylabel(r'$H(t)$', fontsize=12)
ax2.set_title('Early warning signal H(t) - 1D Quintic NLS', fontsize=14)
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('quintic_nls_H_new.png', dpi=200)
plt.show()
print("New image saved: quintic_nls_H_new.png")
