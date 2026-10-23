"""xi = 1.5: stability of the bound and step-size check of the reference (Table 7)."""
import _setup  # noqa: F401
# xi=1.5 diagnostics: (a) stability of the Fourier bound; (b) the gap for several QE step sizes
import numpy as np, json, time, sys
from fk_bound import FKBound
from mc_fk import run_gap
p = dict(S0=100., r=0.04, q=0., T=1., V0=0.09, alpha=1., beta=0.09, gamma=1.5, rho=-0.3)
Ks = [90., 100., 110.]
out = {'a': [], 'b': []}
base = FKBound(**p); lam = {K: base.MLB(K)[1] for K in Ks}
for (d, Nu, Nt, U) in [(1.0,1024,48,None),(0.5,1024,48,None),(0.25,1024,48,None),(1.0,2048,48,None),(1.0,1024,96,None),(1.0,2048,48,'x2')]:
    kw = dict(delta=d, Nu=Nu, Nt=Nt)
    if U == 'x2': kw['U'] = 2*base.U
    fb = FKBound(**p, **kw)
    vals = {K: fb.LB(lam[K], K) for K in Ks}
    out['a'].append(dict(cfg=[d,Nu,Nt,U], LB=vals)); print('a', d, Nu, Nt, U, {K: f"{v:.8f}" for K, v in vals.items()}, flush=True)
json.dump(out, open('diag_xi15.json','w'), indent=1, default=str)
for nst in [64, 1024]:
    t0 = time.time()
    g = run_gap(p, Ks, 4_000_000, nst, 900000 + nst, lam)
    rec = dict(nst=nst, E8={K: out['a'][0]['LB'][K] + g[K][0] for K in Ks}, se={K: g[K][1] for K in Ks})
    out['b'].append(rec); json.dump(out, open('diag_xi15.json','w'), indent=1, default=str)
    print('b', nst, {K: f"{rec['E8'][K]:.6f}±{rec['se'][K]:.6f}" for K in Ks}, f"{time.time()-t0:.0f}s", flush=True)
