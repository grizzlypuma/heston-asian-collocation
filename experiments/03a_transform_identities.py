"""Checks of the two-stage Riccati transform (Section 4.2, Appendix A)."""
import _setup  # noqa: F401
import numpy as np, time
from fk_bound import phi_both
import geo_chf as g

sets = {
 'FK-1': dict(T=1., r=0.04, q=0., V0=0.0483, alpha=4.75, beta=0.0483, gamma=0.55, rho=-0.569),
 'FK-2': dict(T=1., r=0.10, q=0., V0=0.1, alpha=3., beta=0.1, gamma=0.2, rho=0.7),
 'FK-3': dict(T=1., r=0.07, q=0., V0=0.075, alpha=7., beta=0.05, gamma=0.3, rho=-0.4),
 'Corsaro': dict(T=1., r=0.04, q=0., V0=0.09, alpha=1., beta=0.09, gamma=1.0, rho=-0.3),
 'Corsaro T=5 q=.02': dict(T=5., r=0.04, q=0.02, V0=0.09, alpha=1., beta=0.09, gamma=1.0, rho=-0.3),
}
tn = np.array([0.0, 0.1, 0.37, 0.5, 0.9, 1.0])
for name, p in sets.items():
    t0 = time.time()
    tt = tn * p['T']
    # (i) υ=0
    PHI, PHIT = phi_both(np.array([0.0+0j]), tt, **p)
    e1 = np.max(np.abs(PHI[:, 0] - np.exp((p['r']-p['q'])*tt)))
    e1b = abs(PHIT[0]-1)
    # (ii) phi_TTT against geo_chf on the complex grid upsilon = u - i*delta
    ups = np.concatenate([np.linspace(0, 60, 25) - 1.0j, np.linspace(0, 60, 25) - 0.3j, np.linspace(0,60,25)+0j])
    _, PHIT = phi_both(ups, tt[:1], **p)
    ref = g.chf(ups, kap=p['alpha'], th=p['beta'], xi=p['gamma'], rh=p['rho'], v0p=p['V0'],
                S0=1.0, r=p['r'], T=p['T'], q=p['q'])
    e2 = np.max(np.abs(PHIT - ref)); e2r = np.max(np.abs(PHIT - ref)/np.maximum(np.abs(ref),1e-300))
    # (iii) t=0: φ_{0,0,T} = S0 φ_TTT
    PHI, PHIT = phi_both(ups, np.array([0.0]), **p)
    e3 = np.max(np.abs(PHI[0] - PHIT))
    print(f"{name:18s}: (i) max|φ(υ=0)-e^(r-q)t|={e1:.1e}, |φT(0)-1|={e1b:.1e} | (ii) max|φT-geo_chf|={e2:.1e} (rel {e2r:.1e}) | (iii) t=0: {e3:.1e}  ({time.time()-t0:.1f}s)")

# (iv) gamma -> 0 against the closed Gaussian formula E[e^{X_t+iυY}]
sig=0.3; T=1.; r=0.04; q=0.0
p = dict(T=T, r=r, q=q, V0=sig**2, alpha=1.0, beta=sig**2, gamma=1e-9, rho=0.0)
ups = np.linspace(0, 60, 31) - 0.7j
tt = np.array([0., 0.2, 0.5, 0.8, 1.0])
PHI, PHIT = phi_both(ups, tt, **p)
mu = r - q - 0.5*sig**2
worst = 0
for j, t in enumerate(tt):
    EX = mu*t; VX = sig**2*t; EY = mu*T/2; VY = sig**2*T/3; C = sig**2*(t - t*t/(2*T))
    ex = np.exp(EX + 1j*ups*EY + 0.5*(VX + 2j*ups*C - ups**2*VY))
    worst = max(worst, np.max(np.abs(PHI[j]-ex)))
exT = np.exp(1j*ups*mu*T/2 - 0.5*ups**2*sig**2*T/3)
print(f"(iv) Gaussian limit: max|φ-exact|={worst:.1e}, max|φT-exact|={np.max(np.abs(PHIT-exT)):.1e}")
