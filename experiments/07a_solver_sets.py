"""Collocation prices for the strike profile or the regimes. Args: strikes|regimes [k]."""
import _setup  # noqa: F401
# Converged solver values (k=23, Nv=61, Richardson over Nt 1600/3200/6400)
import json, sys, os, time
from arith_order import solve_multi
job = sys.argv[1]; OUT = f'sol_{job}.json'
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
base = dict(r=0.04, T=1.0, kappa=1.0, theta=0.09, xi_=1.0, rho=-0.3, v0=0.09)
regimes = {}
for rho in [-0.9, -0.7, 0.0, 0.5, 0.9]: regimes[f'rho={rho}'] = {**base, 'rho': rho}
for xi in [0.3, 0.6, 1.5]: regimes[f'xi={xi}'] = {**base, 'xi_': xi}
for T in [0.25, 0.5, 3.0, 5.0]: regimes[f'T={T}'] = {**base, 'T': T}
if job == 'strikes':
    todo = {'base': (base, (80., 85., 90., 95., 100., 105., 110., 115., 120., 125., 130.))}
else:
    todo = {k: (v, (90., 100., 110.)) for k, v in regimes.items()}
k = int(sys.argv[2]) if len(sys.argv) > 2 else 23
for name, (p, Ks) in todo.items():
    if name in res: continue
    dom = dict(x_lo=-2.0, x_hi=8.0, v_hi=3.0) if p['T'] >= 3 else dict(x_lo=-1.0, x_hi=5.0, v_hi=1.5)
    t0 = time.time(); P = []
    for Nt in [1600, 3200, 6400]:
        P.append(solve_multi(k, Nv=61, Nt=Nt, Ks=Ks, **dom, **p)[0])
    out = {}
    for K in Ks:
        a, b, c = (x[K] for x in P); r1a = (4*b-a)/3; r1b = (4*c-b)/3
        out[str(K)] = dict(R2=(8*r1b - r1a)/7, raw6400=c, rich_resid=abs((8*r1b-r1a)/7 - r1b))
    res[name] = dict(params=p, dom=dom, k=k, sec=time.time()-t0, K=out)
    json.dump(res, open(OUT, 'w'), indent=1)
    print(name, {K: f"{v['R2']:.7f}" for K, v in out.items()}, f"{time.time()-t0:.0f}s", flush=True)
