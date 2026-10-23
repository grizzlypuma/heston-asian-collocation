"""QE Monte Carlo on the Table 9 sets: price and the bound itself. Args: n_paths n_steps."""
import _setup  # noqa: F401
# Independent MC on the FK Table 9 sets: the price and the bound LB(lambda*) itself (QE + exact geometric CV from geo_chf)
import numpy as np, time, json, sys
from mc_check import qe_paths
from geo_chf import price_carr_madan
from fk_bound import FKBound
sets = {
 'FK-1': dict(S0=100., r=0.04, q=0., T=1., V0=0.0483, alpha=4.75, beta=0.0483, gamma=0.55, rho=-0.569),
 'FK-2': dict(S0=100., r=0.10, q=0., T=1., V0=0.1, alpha=3., beta=0.1, gamma=0.2, rho=0.7),
 'FK-3': dict(S0=100., r=0.07, q=0., T=1., V0=0.075, alpha=7., beta=0.05, gamma=0.3, rho=-0.4),
}
K = 100.; n_paths = int(sys.argv[1]) if len(sys.argv)>1 else 2_000_000; nst = int(sys.argv[2]) if len(sys.argv)>2 else 256
out = {}
for name, p in sets.items():
    t0 = time.time()
    fb = FKBound(**p); mlb, lam = fb.MLB(K)
    geo = price_carr_madan(K, kap=p['alpha'], th=p['beta'], xi=p['gamma'], rh=p['rho'], v0p=p['V0'],
                           S0=p['S0'], r=p['r'], T=p['T'], q=p['q'])
    disc = np.exp(-p['r']*p['T'])
    acc = {k: [] for k in ['y','l','c']}
    b = 0; done = 0; bs = 250_000
    while done < n_paths:
        A, G = qe_paths(bs, nst, p['alpha'], p['beta'], p['gamma'], p['rho'], p['V0'], p['S0'], p['r'], p['T'], 9000+b, q=p['q'])
        acc['y'].append(disc*np.maximum(A-K, 0)); acc['l'].append(disc*(A-K)*(np.log(G) > lam)); acc['c'].append(disc*np.maximum(G-K,0) - geo)
        done += bs; b += 1
    y, l, c = (np.concatenate(acc[k]) for k in ['y','l','c'])
    def cv(z):
        cz = np.cov(z, c); beta = cz[0,1]/cz[1,1]; e = z - beta*c
        return e.mean(), e.std(ddof=1)/np.sqrt(len(e)), z.std(ddof=1)/e.std(ddof=1)
    py, sy, gy = cv(y); pl, sl, gl = cv(l)
    # price minus LB on the same paths (directly: the remainder)
    d = y - l; pd, sd = d.mean(), d.std(ddof=1)/np.sqrt(len(d))
    out[name] = dict(mlb=mlb, lam=lam, price=py, price_se=sy, lb_mc=pl, lb_se=sl, gap_mc=pd, gap_se=sd, geo=geo, n=n_paths, nst=nst)
    print(f"{name}: MC price {py:.5f}±{sy:.5f} (gain {gy:.0f}x) | MC LB(λ*) {pl:.5f}±{sl:.5f} vs our MLB {mlb:.5f} ({(pl-mlb)/sl:+.2f}σ) | price−LB on paths {pd:.5f}±{sd:.5f} ({time.time()-t0:.0f}s)", flush=True)
json.dump(out, open(f'mcB3_{n_paths}_{nst}.json','w'), indent=1)
