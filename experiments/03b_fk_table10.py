"""Reproduction of Table 10 of Fusai and Kyriakou (2016), Gaussian limit (Table 1)."""
import _setup  # noqa: F401
import numpy as np, time, json
from fk_bound import FKBound, lb_gauss, mlb_gauss
cases = [(2,2,0.02,0.1,1,0.055985,2.00),(2,2,0.18,0.3,1,0.218366,1.99),(2,2,0.0125,0.25,2,0.172226,1.99),
         (1.9,2,0.05,0.5,1,0.193060,1.97),(2,2,0.05,0.5,1,0.246298,1.98),(2.1,2,0.05,0.5,1,0.306094,1.97),
         (2,2,0.05,0.5,2,0.349779,1.95)]
out=[]
print(f"{'S0':>4}{'r':>7}{'σ':>5}{'T':>3} | {'MLB closed':>12} {'MLB Fourier':>12} {'Table10':>9} {'diff':>9} | e^λ* | max|LBfour-LBcl| over λ")
for (S0,K,r,s,T,tab,elam) in cases:
    t0=time.time()
    mc, lc = mlb_gauss(S0,K,r,0.0,s,T)
    fb = FKBound(S0, r, 0.0, T, s*s, 1.0, s*s, 1e-9, 0.0, delta=1.0)
    mf, lf = fb.MLB(K)
    lams = np.log(K) + np.linspace(-0.3, 0.3, 13)
    dmax = max(abs(fb.LB(l,K)-lb_gauss(l,S0,K,r,0.0,s,T)) for l in lams)
    print(f"{S0:4.1f}{r:7.4f}{s:5.2f}{T:3.0f} | {mc:12.8f} {mf:12.8f} {tab:9.6f} {mf-tab:+9.1e} | {np.exp(lf):.3f} ({elam}) | {dmax:.1e}  ({time.time()-t0:.1f}s)")
    out.append(dict(S0=S0,K=K,r=r,sigma=s,T=T,mlb_closed=mc,mlb_fourier=mf,table10=tab,elam=np.exp(lf),elam_tab=elam,dmax=dmax))
json.dump(out, open('B2.json','w'), indent=1)
