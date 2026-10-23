"""Seven alternative parameter readings for Table 9 (Section 5.6)."""
import _setup  # noqa: F401
import numpy as np, time, json
from fk_bound import FKBound
base = {
 'FK-1': (dict(S0=100., r=0.04, q=0., T=1., V0=0.0483, alpha=4.75, beta=0.0483, gamma=0.55, rho=-0.569), 5.854),
 'FK-2': (dict(S0=100., r=0.10, q=0., T=1., V0=0.1, alpha=3., beta=0.1, gamma=0.2, rho=0.7), 9.323),
 'FK-3': (dict(S0=100., r=0.07, q=0., T=1., V0=0.075, alpha=7., beta=0.05, gamma=0.3, rho=-0.4), 7.123),
}
variants = {
 'as printed': lambda p: p,
 'rho -> -rho': lambda p: {**p, 'rho': -p['rho']},
 'gamma -> sqrt(gamma)': lambda p: {**p, 'gamma': np.sqrt(p['gamma'])},
 'gamma -> gamma^2': lambda p: {**p, 'gamma': p['gamma']**2},
 'V0, beta read as vol (squared)': lambda p: {**p, 'V0': p['V0']**2, 'beta': p['beta']**2},
 'V0 read as vol (squared)': lambda p: {**p, 'V0': p['V0']**2},
 'rho=0': lambda p: {**p, 'rho': 0.0},
 'gamma=0 (deterministic V)': lambda p: {**p, 'gamma': 1e-9, 'rho': 0.0},
}
out = {}
for vn, f in variants.items():
    row = []
    for name,(p,tab) in base.items():
        fb = FKBound(**f(p), Nu=768, Nt=40)
        m,_ = fb.MLB(100.)
        row.append(m - tab)
    out[vn] = row
    print(f"{vn:24s}: MLB_ours - Table9 = " + "  ".join(f"{d:+.4f}" for d in row), flush=True)
json.dump(out, open('diagB3.json','w'), indent=1)
