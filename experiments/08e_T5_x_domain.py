"""Which side of the x-domain matters at T = 5 (Table 7)."""
import _setup  # noqa: F401
# T=5: which side of the x-domain matters, and whether the price reaches a plateau
import json, time, os, subprocess
from arith_order import solve_multi
OUT = 'diag_T5_x.json'; res = json.load(open(OUT)) if os.path.exists(OUT) else {}
Ks = (90., 100., 110.); p = dict(r=0.04, T=5.0, kappa=1.0, theta=0.09, xi_=1.0, rho=-0.3, v0=0.09)
for tag, k, dom in [('x_lo=-3,x_hi=8', 23, dict(x_lo=-3., x_hi=8., v_hi=3.)), ('x_lo=-2,x_hi=12', 23, dict(x_lo=-2., x_hi=12., v_hi=3.)),
                    ('[-3,12] k=32', 32, dict(x_lo=-3., x_hi=12., v_hi=3.)), ('[-4,16] k=32', 32, dict(x_lo=-4., x_hi=16., v_hi=3.))]:
    if tag in res: continue
    t0 = time.time(); P = []
    for Nt in [1600, 3200, 6400]:
        P.append(solve_multi(k, Nv=61, Nt=Nt, Ks=Ks, **dom, **p)[0])
    out = {}
    for K in Ks:
        a, b, c = (x[K] for x in P); r1a = (4*b-a)/3; r1b = (4*c-b)/3
        out[str(K)] = (8*r1b - r1a)/7
    res[tag] = dict(k=k, dom=dom, K=out, sec=time.time()-t0); json.dump(res, open(OUT, 'w'), indent=1)
    print(tag, {K: f"{v:.7f}" for K, v in out.items()}, f"{time.time()-t0:.0f}s", flush=True)
