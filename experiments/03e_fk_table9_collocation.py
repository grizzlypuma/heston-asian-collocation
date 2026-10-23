"""Collocation prices for the three Heston sets of Table 9 of Fusai and Kyriakou (2016) (Table 8).

k=23 (Nx=392), Nv=61, two-level Richardson over Nt=1600/3200/6400. Arg: base|vhi3 selects the
domain: x in [-1,5], v_max=1.5 (the default) or the same with v_max=3 as a truncation check.
Writes fk_table9_colloc_<arg>.json (entries are added, finished ones are skipped).
"""
import _setup  # noqa: F401
import json, os, sys, time
from arith_order import solve_multi

sets = {
    'FK-1': dict(r=0.04, T=1.0, kappa=4.75, theta=0.0483, xi_=0.55, rho=-0.569, v0=0.0483),
    'FK-2': dict(r=0.10, T=1.0, kappa=3.0, theta=0.1, xi_=0.2, rho=0.7, v0=0.1),
    'FK-3': dict(r=0.07, T=1.0, kappa=7.0, theta=0.05, xi_=0.3, rho=-0.4, v0=0.075),
}
which = sys.argv[1] if len(sys.argv) > 1 else 'base'
dom = dict(x_lo=-1.0, x_hi=5.0, v_hi=1.5 if which == 'base' else 3.0)
OUT = f'fk_table9_colloc_{which}.json'
K = 100.0
for name, p in sets.items():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    tag = f'{name}|{which}'
    if tag in res:
        continue
    t0 = time.time()
    P = [solve_multi(23, Nv=61, Nt=Nt, Ks=(K,), **dom, **p)[0][K] for Nt in (1600, 3200, 6400)]
    r1a, r1b = (4 * P[1] - P[0]) / 3, (4 * P[2] - P[1]) / 3
    r2 = (8 * r1b - r1a) / 7
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res[tag] = dict(params=p, dom=dom, k=23, Nv=61, R2=r2, raw=P, rich_resid=abs(r2 - r1b), sec=time.time() - t0)
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f"{tag}: {r2:.7f}  (raw Nt=6400 {P[2]:.7f}, {time.time() - t0:.0f}s)", flush=True)
