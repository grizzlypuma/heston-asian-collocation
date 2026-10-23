"""Reference price MLB + E[D] for the base set. Args: n_paths steps(comma) out.json"""
import _setup  # noqa: F401
# Reference: the gap (price - LB_FK) on QE paths for several n_steps; only run_gap is needed (cheap)
import numpy as np, json, time, sys
from fk_bound import FKBound
from mc_fk import run_gap
p = dict(S0=100., r=0.04, q=0., T=1., V0=0.09, alpha=1., beta=0.09, gamma=1.0, rho=-0.3)
Ks = [90., 100., 110.]
fb = FKBound(**p); mlb = {}; lam = {}
for K in Ks: mlb[K], lam[K] = fb.MLB(K)
n_paths = int(sys.argv[1]); steps = [int(s) for s in sys.argv[2].split(',')]; OUTF = sys.argv[3]
out = {'mlb': {str(K): mlb[K] for K in Ks}, 'lam': {str(K): lam[K] for K in Ks}, 'runs': []}
for nst in steps:
    t0 = time.time()
    g = run_gap(p, Ks, n_paths, nst, 50000 + 1000*nst, lam)
    rec = dict(nst=nst, n=n_paths, sec=time.time()-t0, gap={str(K): g[K][0] for K in Ks}, se={str(K): g[K][1] for K in Ks},
               E8={str(K): mlb[K] + g[K][0] for K in Ks})
    out['runs'].append(rec); json.dump(out, open(OUTF, 'w'), indent=1)
    print(nst, {K: f"{mlb[K]+g[K][0]:.6f}±{g[K][1]:.6f}" for K in Ks}, f"{rec['sec']:.0f}s", flush=True)
