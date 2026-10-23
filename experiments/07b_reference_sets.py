"""Reference prices for the strike profile or the regimes. Args: strikes|regimes n_paths n_steps."""
import _setup  # noqa: F401
# Reference values (MLB + MC gap) for the parameter regimes and the strike profile
import numpy as np, json, time, sys, os
from fk_bound import FKBound
from mc_fk import run_gap
base = dict(S0=100., r=0.04, q=0., T=1., V0=0.09, alpha=1., beta=0.09, gamma=1.0, rho=-0.3)
regimes = {}
for rho in [-0.9, -0.7, 0.0, 0.5, 0.9]: regimes[f'rho={rho}'] = {**base, 'rho': rho}
for xi in [0.3, 0.6, 1.5]: regimes[f'xi={xi}'] = {**base, 'gamma': xi}
for T in [0.25, 0.5, 3.0, 5.0]: regimes[f'T={T}'] = {**base, 'T': T}
job = sys.argv[1]; n_paths = int(sys.argv[2]); nst = int(sys.argv[3])
OUT = f'e8_{job}.json'
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
if job == 'strikes':
    todo = {'base': (base, [80., 85., 90., 95., 100., 105., 110., 115., 120., 125., 130.])}
else:
    todo = {k: (v, [90., 100., 110.]) for k, v in regimes.items()}
for name, (p, Ks) in todo.items():
    tag = f'{name}|{nst}'
    if tag in res: continue
    t0 = time.time()
    for delta in [1.0, 0.5, 0.25]:
        try:
            fb = FKBound(**p, delta=delta)
            mlb = {}; lam = {}
            for K in Ks: mlb[K], lam[K] = fb.MLB(K)
            if all(np.isfinite(list(mlb.values()))): break
        except Exception as ex:
            print(name, 'delta', delta, 'fail:', ex, flush=True)
    g = run_gap(p, Ks, n_paths, nst, 300000 + 7919*len(res), lam)
    res[tag] = dict(params=p, delta=delta, n=n_paths, nst=nst, sec=time.time()-t0,
                    K={str(K): dict(mlb=mlb[K], lam=lam[K], gap=g[K][0], gap_se=g[K][1], E8=mlb[K]+g[K][0]) for K in Ks})
    json.dump(res, open(OUT, 'w'), indent=1)
    print(name, nst, {K: f"{mlb[K]+g[K][0]:.6f}±{g[K][1]:.6f} (gap {g[K][0]:.5f})" for K in Ks}, f"{time.time()-t0:.0f}s", flush=True)
