"""Monte Carlo with exact geometric control variate at T = 0.25, 3, 5 (Section 4.1)."""
import _setup  # noqa: F401
import numpy as np, time, json
import mc_check as mc
sol = {0.25: 3.539285, 3.0: 12.535696, 5.0: 16.549966}
out = {}
for T in [0.25, 3.0, 5.0]:
    t0=time.time()
    m, se, corr, gain, d = mc.mc_price(K=100., T=T, n_paths=2_000_000, seed=777, diag=True)
    print(f"T={T}: CV {m:.6f}±{se:.6f} corr {corr:.4f} gain {gain:.1f}x | E[c]={d['c_mean']:+.6f}±{d['c_se']:.6f} ({d['c_mean']/d['c_se']:+.2f}σ) | solver {sol[T]} -> {(sol[T]-m)/se:+.2f}σ ({time.time()-t0:.0f}s)")
    out[T] = dict(cv=m, se=se, corr=corr, gain=gain, **d, solver=sol[T])
json.dump(out, open('A4b.json','w'), indent=1)
