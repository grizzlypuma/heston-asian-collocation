"""Full-truncation Euler Monte Carlo on the Table 9 sets, with and without control variates (Table 8).
Args: sets(comma) n_paths steps(comma) [append]"""
import _setup  # noqa: F401
import numpy as np, json, time, sys
from geo_chf import price_carr_madan
from fk_bound import FKBound
from mc_euler import ft_euler

sets = {
 'FK-1': dict(S0=100., r=0.04, q=0., T=1., V0=0.0483, alpha=4.75, beta=0.0483, gamma=0.55, rho=-0.569),
 'FK-2': dict(S0=100., r=0.10, q=0., T=1., V0=0.1, alpha=3., beta=0.1, gamma=0.2, rho=0.7),
 'FK-3': dict(S0=100., r=0.07, q=0., T=1., V0=0.075, alpha=7., beta=0.05, gamma=0.3, rho=-0.4),
}
names = sys.argv[1].split(','); n_paths = int(sys.argv[2]); steps = [int(s) for s in sys.argv[3].split(',')]
K = 100.; bs = 200_000
out = json.load(open('mc_euler.json')) if len(sys.argv) > 4 and sys.argv[4] == 'append' else {}
for name in names:
    p = sets[name]
    fb = FKBound(**p); mlb, lam = fb.MLB(K)
    geo = price_carr_madan(K, kap=p['alpha'], th=p['beta'], xi=p['gamma'], rh=p['rho'], v0p=p['V0'], S0=p['S0'], r=p['r'], T=p['T'], q=p['q'])
    disc = np.exp(-p['r']*p['T'])
    for nst in steps:
        t0 = time.time(); m = np.zeros(8); b = 0; done = 0
        while done < n_paths:
            A, Y = ft_euler(bs, nst, p['alpha'], p['beta'], p['gamma'], p['rho'], p['V0'], p['S0'], p['r'], p['T'], 700000 + 1000*nst + b, q=p['q'])
            y = disc*np.maximum(A-K, 0); c = disc*np.maximum(np.exp(Y)-K, 0) - geo
            d = y - disc*(A-K)*(Y > lam)
            m += [len(y), y.sum(), (y*y).sum(), c.sum(), (c*c).sum(), (y*c).sum(), d.sum(), (d*d).sum()]
            done += bs; b += 1
        n, sy, syy, sc, scc, syc, sd, sdd = m
        my, mc, md = sy/n, sc/n, sd/n
        vy, vc, cyc = syy/n-my**2, scc/n-mc**2, syc/n-my*mc
        beta = cyc/vc; pcv = my - beta*mc; se_cv = np.sqrt((vy - beta*cyc)/n)
        rec = dict(nst=nst, n=int(n), raw=my, raw_se=np.sqrt(vy/n), cv=pcv, cv_se=se_cv, c_mean=mc, c_se=np.sqrt(vc/n),
                   gap=md, gap_se=np.sqrt((sdd/n-md**2)/n), mlb=mlb, sec=time.time()-t0)
        out.setdefault(name, []).append(rec); json.dump(out, open('mc_euler.json', 'w'), indent=1)
        print(f"{name} nst={nst}: raw {my:.5f}±{rec['raw_se']:.5f} | geom.CV {pcv:.5f}±{se_cv:.5f} | E[c]={mc:+.5f}±{rec['c_se']:.5f} | MLB+gap {mlb+md:.5f}±{rec['gap_se']:.5f} ({rec['sec']:.0f}s)", flush=True)
