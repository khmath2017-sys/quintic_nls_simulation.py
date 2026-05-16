import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def S_rhs(t, S, A, Tc):
    dSdt = A / (2 * (Tc - t) ** 1.5)
    return dSdt

Tc = 1.0
A = 1.0
t_span = (0, 0.99 * Tc)
t_eval = np.linspace(0, 0.99 * Tc, 500)
S0 = [A / np.sqrt(Tc)]

sol = solve_ivp(S_rhs, t_span, S0, t_eval=t_eval, args=(A, Tc))
t = sol.t
S = sol.y[0]

ell = 1.0
dS = A / (2 * (Tc - t) ** 1.5)
H = ell * dS / (S ** 2 + 1e-12)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
ax1.plot(t, S, 'b-', linewidth=2)
ax1.set_xlabel('Time t')
ax1.set_ylabel('S(t)')
ax1.set_title('Sobolev-Stepanov norm evolution')
ax1.grid(True)

ax2.plot(t, H, 'r-', linewidth=2, label='H(t)')
ax2.axhline(y=5, color='gold', linestyle='--', label='Warning threshold')
ax2.set_xlabel('Time t')
ax2.set_ylabel('H(t)')
ax2.set_title('Early warning signal H(t) - 1D Quintic NLS')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('quintic_nls_H.png', dpi=150)
plt.show()
print("Image saved as quintic_nls_H.png")
