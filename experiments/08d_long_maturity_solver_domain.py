"""Domain tests at T = 3, 5 (Table 7)."""
import _setup  # noqa: F401
# Long maturities, solver side: change one domain parameter at a time. Base: k=23, x in [-2,8], v_hi=3
import json, time, os
from arith_order import solve_multi
OUT = 'diag_longT_solver.json'; res = json.load(open(OUT)) if os.path.exists(OUT) else {}
Ks = (90., 100., 110.)
for T in [3.0, 5.0]:
    p = dict(r=0.04, T=T, kappa=1.0, theta=0.09, xi_=1.0, rho=-0.3, v0=0.09)
    for name, dom in [('v_hi=5', dict(x_lo=-2., x_hi=8., v_hi=5.)), ('x∈[-3,12]', dict(x_lo=-3., x_hi=12., v_hi=3.)),
                      ('v_hi=8', dict(x_lo=-2., x_hi=8., v_hi=8.))]:
        tag = f'T={T} {name}'
        if tag in res: continue
        t0 = time.time(); P = []
        for Nt in [1600, 3200, 6400]:
            P.append(solve_multi(23, Nv=61, Nt=Nt, Ks=Ks, **dom, **p)[0])
        out = {}
        for K in Ks:
            a, b, c = (x[K] for x in P); r1a = (4*b-a)/3; r1b = (4*c-b)/3
            out[str(K)] = (8*r1b - r1a)/7
        res[tag] = dict(dom=dom, K=out, sec=time.time()-t0); json.dump(res, open(OUT, 'w'), indent=1)
        print(tag, {K: f"{v:.7f}" for K, v in out.items()}, f"{time.time()-t0:.0f}s", flush=True)
