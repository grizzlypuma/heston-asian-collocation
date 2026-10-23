"""rho = 0.9: finer x-grid and more reference runs. Args: solver|mc."""
import _setup  # noqa: F401
import json, time, sys
which = sys.argv[1]
if which == 'solver':
    from arith_order import solve_multi
    p = dict(r=0.04, T=1.0, kappa=1.0, theta=0.09, xi_=1.0, rho=0.9, v0=0.09); Ks = (90., 100., 110.)
    out = {}
    for tag, k, Nv in [('k=32', 32, 61), ('Nv=91', 23, 91)]:
        t0 = time.time(); P = [solve_multi(k, Nv=Nv, Nt=Nt, Ks=Ks, x_lo=-1., x_hi=5., v_hi=3., **p)[0] for Nt in [1600, 3200, 6400]]
        out[tag] = {}
        for K in Ks:
            a, b, c = (x[K] for x in P); r1a = (4*b-a)/3; r1b = (4*c-b)/3; out[tag][str(K)] = (8*r1b-r1a)/7
        json.dump(out, open('diag_rho09_solver.json', 'w'), indent=1)
        print(tag, {K: f"{v:.7f}" for K, v in out[tag].items()}, f"{time.time()-t0:.0f}s", flush=True)
else:
    from fk_bound import FKBound
    from mc_fk import run_gap
    p = dict(S0=100., r=0.04, q=0., T=1., V0=0.09, alpha=1., beta=0.09, gamma=1.0, rho=0.9); Ks = [90., 100., 110.]
    fb = FKBound(**p); mlb = {}; lam = {}
    for K in Ks: mlb[K], lam[K] = fb.MLB(K)
    out = {}
    for nst in [64, 1024]:
        t0 = time.time(); g = run_gap(p, Ks, 4_000_000, nst, 1500000 + nst, lam)
        out[str(nst)] = {str(K): dict(E8=mlb[K]+g[K][0], se=g[K][1]) for K in Ks}
        json.dump(out, open('diag_rho09_mc.json', 'w'), indent=1)
        print(nst, {K: f"{mlb[K]+g[K][0]:.6f}±{g[K][1]:.6f}" for K in Ks}, f"{time.time()-t0:.0f}s", flush=True)
