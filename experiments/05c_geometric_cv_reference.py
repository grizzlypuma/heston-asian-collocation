"""Reference of the earlier kind: QE paths with the exact geometric price as the only control variate
(Kemna-Vorst), for comparison with the reference of Table 2. Args: [n_paths] [n_steps]
(default 8,000,000 paths, 512 steps). Writes geo_cv_reference.json.
"""
import _setup  # noqa: F401
import json, sys, time
import numpy as np
from mc_check import qe_paths
from geo_chf import price_carr_madan

n_paths = int(sys.argv[1]) if len(sys.argv) > 1 else 8_000_000
nst = int(sys.argv[2]) if len(sys.argv) > 2 else 512
p = dict(kappa=1.0, theta=0.09, xi_=1.0, rho=-0.3, v0=0.09, S0=100.0, r=0.04, T=1.0)
Ks = [90.0, 100.0, 110.0]
disc = np.exp(-p['r'] * p['T'])
geo = {K: price_carr_madan(K, kap=p['kappa'], th=p['theta'], xi=p['xi_'], rh=p['rho'], v0p=p['v0'],
                           S0=p['S0'], r=p['r'], T=p['T']) for K in Ks}
S = {K: np.zeros(5) for K in Ks}          # sums of y, y^2, c, c^2, c*y
t0 = time.time(); done = 0; b = 0; batch = 250_000
while done < n_paths:
    m = min(batch, n_paths - done)
    A, G = qe_paths(m, nst, p['kappa'], p['theta'], p['xi_'], p['rho'], p['v0'], p['S0'], p['r'], p['T'],
                    seed=910000 + b)
    for K in Ks:
        y = disc * np.maximum(A - K, 0.0)
        c = disc * np.maximum(G - K, 0.0) - geo[K]
        S[K] += [y.sum(), (y * y).sum(), c.sum(), (c * c).sum(), (c * y).sum()]
    done += m; b += 1
out = {}
for K in Ks:
    sy, syy, sc, scc, scy = S[K] / n_paths
    vy, vc, cov = syy - sy ** 2, scc - sc ** 2, scy - sy * sc
    beta = cov / vc
    price = sy - beta * sc
    se = np.sqrt((vy - 2 * beta * cov + beta ** 2 * vc) / n_paths)
    out[str(K)] = dict(price=price, se=se, raw_se=np.sqrt(vy / n_paths), corr=cov / np.sqrt(vy * vc),
                       geo_exact=geo[K])
    print(f"K={K:5.0f}: {price:.6f} ± {se:.6f}  (raw s.e. {np.sqrt(vy / n_paths):.5f}, corr {cov / np.sqrt(vy * vc):.4f})")
out['meta'] = dict(n=n_paths, nst=nst, sec=time.time() - t0)
json.dump(out, open('geo_cv_reference.json', 'w'), indent=1)
print(f"{time.time() - t0:.0f}s")
