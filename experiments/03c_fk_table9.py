"""Maximal lower bound for the Heston sets of Table 9 of Fusai and Kyriakou (2016) (Table 8)."""
import _setup  # noqa: F401
import numpy as np, time, json, os
from fk_bound import FKBound
# collocation prices from 03e_fk_table9_collocation.py (NaN if that script has not been run)
colloc = json.load(open('fk_table9_colloc_base.json')) if os.path.exists('fk_table9_colloc_base.json') else {}
sol = lambda name: colloc[f'{name}|base']['R2'] if f'{name}|base' in colloc else float('nan')
sets = {
 'FK-1': (dict(S0=100., r=0.04, q=0., T=1., V0=0.0483, alpha=4.75, beta=0.0483, gamma=0.55, rho=-0.569), 5.854, sol('FK-1')),
 'FK-2': (dict(S0=100., r=0.10, q=0., T=1., V0=0.1, alpha=3., beta=0.1, gamma=0.2, rho=0.7), 9.323, sol('FK-2')),
 'FK-3': (dict(S0=100., r=0.07, q=0., T=1., V0=0.075, alpha=7., beta=0.05, gamma=0.3, rho=-0.4), 7.123, sol('FK-3')),
}
out = {}
K = 100.
for name,(p,tab,solv) in sets.items():
    t0=time.time()
    res = {}
    for (d,Nt,Nu) in [(1.0,48,1024),(0.5,48,1024),(2.0,48,1024),(1.0,64,1024),(1.0,48,2048)]:
        fb = FKBound(**p, delta=d, Nt=Nt, Nu=Nu)
        m, l = fb.MLB(K)
        res[(d,Nt,Nu)] = (m,l)
    fb = FKBound(**p)
    m, l = res[(1.0,48,1024)]
    slb = fb.LB(np.log(K), K)
    spread = max(v[0] for v in res.values()) - min(v[0] for v in res.values())
    print(f"{name}: MLB={m:.7f} (Table9 {tab}, diff {m-tab:+.1e}) e^λ*={np.exp(l):.3f} | SLB(lnK)={slb:.7f} (MLB-SLB={m-slb:.1e}) | spread over δ/Nt/Nu {spread:.1e} | solver {solv:.7f} gap {solv-m:+.5f} ({time.time()-t0:.0f}s)")
    out[name] = dict(mlb=m, elam=float(np.exp(l)), slb=slb, table9=tab, solver=solv, spread=spread)
json.dump(out, open('B3.json','w'), indent=1)
