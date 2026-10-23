# Monte Carlo with the Fusai-Kyriakou bound as control variate (Section 4.3):
# price = MLB (exact for the continuous average) + E[D],  D = payoff - bound payoff on each path.
# run() also reports the estimate with the exact geometric control variate.
import numpy as np, time, json, sys
from mc_check import qe_paths
from geo_chf import price_carr_madan
from fk_bound import FKBound

def run(p, Ks, n_paths, nst, seed0, bs=250_000):
    fb = FKBound(**p)
    mlb = {}; lam = {}; geo = {}
    for K in Ks:
        mlb[K], lam[K] = fb.MLB(K)
        geo[K] = price_carr_madan(K, kap=p['alpha'], th=p['beta'], xi=p['gamma'], rh=p['rho'], v0p=p['V0'],
                                  S0=p['S0'], r=p['r'], T=p['T'], q=p['q'])
    disc = np.exp(-p['r']*p['T'])
    S = {K: np.zeros(9) for K in Ks}   # n, Σy, Σy², Σl, Σl², Σc, Σc², Σyc, Σlc
    done = 0; b = 0
    while done < n_paths:
        A, G = qe_paths(bs, nst, p['alpha'], p['beta'], p['gamma'], p['rho'], p['V0'], p['S0'], p['r'], p['T'], seed0+b, q=p['q'])
        Y = np.log(G)
        for K in Ks:
            y = disc*np.maximum(A-K, 0); l = disc*(A-K)*(Y > lam[K]); c = disc*np.maximum(G-K, 0) - geo[K]
            S[K] += [len(y), y.sum(), (y*y).sum(), l.sum(), (l*l).sum(), c.sum(), (c*c).sum(), (y*c).sum(), (l*c).sum()]
        done += bs; b += 1
    out = {}
    for K in Ks:
        n, sy, syy, sl, sll, sc, scc, syc, slc = S[K]
        my, ml, mc = sy/n, sl/n, sc/n
        vy, vl, vc = syy/n-my**2, sll/n-ml**2, scc/n-mc**2
        cyc, clc, cyl = syc/n-my*mc, slc/n-ml*mc, None
        by, bl = cyc/vc, clc/vc
        py = my - by*mc; sey = np.sqrt((vy - by*cyc)/n)
        pl = ml - bl*mc; sel = np.sqrt((vl - bl*clc)/n)
        # remainder d = y - l: E[d] and its s.e. are obtained in run_gap on the same paths
        out[K] = dict(mlb=mlb[K], lam=lam[K], price_geoCV=py, se_geoCV=sey, lb_mc=pl, lb_se=sel)
    return out, fb, lam, disc

def run_gap(p, Ks, n_paths, nst, seed0, lam, bs=250_000):
    disc = np.exp(-p['r']*p['T'])
    S = {K: np.zeros(3) for K in Ks}
    done = 0; b = 0
    while done < n_paths:
        A, G = qe_paths(bs, nst, p['alpha'], p['beta'], p['gamma'], p['rho'], p['V0'], p['S0'], p['r'], p['T'], seed0+b, q=p['q'])
        Y = np.log(G)
        for K in Ks:
            d = disc*(np.maximum(A-K, 0) - (A-K)*(Y > lam[K]))
            S[K] += [len(d), d.sum(), (d*d).sum()]
        done += bs; b += 1
    return {K: (S[K][1]/S[K][0], np.sqrt((S[K][2]/S[K][0] - (S[K][1]/S[K][0])**2)/S[K][0])) for K in Ks}

if __name__ == '__main__':
    tag = sys.argv[1]; n_paths = int(sys.argv[2]); nst = int(sys.argv[3]); seed0 = int(sys.argv[4])
    P = {'corsaro': dict(S0=100., r=0.04, q=0., T=1., V0=0.09, alpha=1., beta=0.09, gamma=1.0, rho=-0.3)}
    p = P[tag]; Ks = [90., 100., 110.]
    t0 = time.time()
    out, fb, lam, disc = run(p, Ks, n_paths, nst, seed0)
    gap = run_gap(p, Ks, n_paths, nst, seed0, lam)   # same seeds -> same paths
    for K in Ks:
        o = out[K]; g, gs = gap[K]
        o['gap'] = g; o['gap_se'] = gs; o['E8'] = o['mlb'] + g
        print(f"K={K:5.0f}: MLB {o['mlb']:.6f} (e^λ*={np.exp(o['lam']):.3f}) | MC LB(λ*) {o['lb_mc']:.6f}±{o['lb_se']:.6f} ({(o['lb_mc']-o['mlb'])/o['lb_se']:+.2f}σ)"
              f" | MC price(geom.CV) {o['price_geoCV']:.6f}±{o['se_geoCV']:.6f} | gap {g:.6f}±{gs:.6f} | reference = MLB+gap {o['E8']:.6f}", flush=True)
    json.dump({str(k): v for k, v in out.items()} | {'meta': dict(n=n_paths, nst=nst, seed0=seed0, sec=time.time()-t0)},
              open(f'mcfk_{tag}_{n_paths}_{nst}_{seed0}.json', 'w'), indent=1)
    print(f"({time.time()-t0:.0f}s)")
