"""Reference at T = 3, 5 with 2048 steps."""
import _setup  # noqa: F401
# Long maturities: reference gap with 2048 steps for T=3 and T=5
import json, time, subprocess
from fk_bound import FKBound
from mc_fk import run_gap
out = {}
for T in [3.0, 5.0]:
    p = dict(S0=100., r=0.04, q=0., T=T, V0=0.09, alpha=1., beta=0.09, gamma=1.0, rho=-0.3)
    Ks = [90., 100., 110.]; fb = FKBound(**p); mlb = {}; lam = {}
    for K in Ks: mlb[K], lam[K] = fb.MLB(K)
    t0 = time.time(); g = run_gap(p, Ks, 4_000_000, 2048, 1300000 + int(T*1000), lam)
    out[str(T)] = {str(K): dict(E8=mlb[K]+g[K][0], se=g[K][1]) for K in Ks}
    json.dump(out, open('diag_longT_2048.json','w'), indent=1)
    print(T, {K: f"{mlb[K]+g[K][0]:.6f}±{g[K][1]:.6f}" for K in Ks}, f"{time.time()-t0:.0f}s", flush=True)
