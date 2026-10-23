"""Collocation prices at T = 3, 5 on three configurations."""
import _setup  # noqa: F401
# Solver re-check at long maturities (T=3,5) against MC with the exact geometric control variate
import json, time, sys
from arith_colloc import solve_arith
cfgs = [
  dict(Nx=181, Nv=41, Nt=1600, x_lo=-2., x_hi=8., v_hi=3.0, ax=0.3),
  dict(Nx=241, Nv=61, Nt=3200, x_lo=-2., x_hi=8., v_hi=3.0, ax=0.3),
  dict(Nx=241, Nv=61, Nt=3200, x_lo=-3., x_hi=12., v_hi=5.0, ax=0.3),
]
out = []
for T in [3.0, 5.0]:
    for c in cfgs:
        t0 = time.time()
        p = solve_arith(K=100., T=T, **c)
        rec = dict(T=T, price=p, sec=time.time()-t0, **c)
        out.append(rec); print(rec, flush=True)
        json.dump(out, open('A5.json', 'w'), indent=1)
