"""xi = 1.5: one change at a time on the solver side (Table 7)."""
import _setup  # noqa: F401
# xi=1.5 diagnostics on the solver side: one parameter at a time, the rest as in the base run (k=23, Nv=61, v_hi=1.5, av=0.15)
import json, time, subprocess, os
from arith_order import solve_multi
p = dict(r=0.04, T=1.0, kappa=1.0, theta=0.09, xi_=1.5, rho=-0.3, v0=0.09)
Ks = (90., 100., 110.)
variants = {'base': {}, 'v_hi=3': dict(v_hi=3.0), 'v_hi=5': dict(v_hi=5.0), 'Nv=91': dict(Nv=91),
            'av=0.05': dict(av=0.05), 'k=32': dict(k=32)}
OUT = 'diag_xi15_solver.json'
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for name, v in variants.items():
    if name in res: continue
    kw = dict(k=23, Nv=61, v_hi=1.5, av=0.15); kw.update(v)
    k = kw.pop('k'); t0 = time.time(); P = []
    for Nt in [1600, 3200, 6400]:
        P.append(solve_multi(k, Nt=Nt, Ks=Ks, **kw, **p)[0])
    out = {}
    for K in Ks:
        a, b, c = (x[K] for x in P); r1a = (4*b-a)/3; r1b = (4*c-b)/3
        out[str(K)] = (8*r1b - r1a)/7
    res[name] = dict(kw=kw, k=k, K=out, sec=time.time()-t0)
    json.dump(res, open(OUT, 'w'), indent=1)
    print(name, {K: f"{v:.7f}" for K, v in out.items()}, f"{time.time()-t0:.0f}s", flush=True)
