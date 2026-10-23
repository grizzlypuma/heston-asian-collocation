"""Spatial order estimates from the convergence runs (Table 6)."""
import _setup  # noqa: F401
import json, sys, numpy as np
from scipy.optimize import brentq

# Two-level Richardson extrapolation in time (Nt=1600/3200/6400),
# then the spatial order: (a) from successive differences (no reference needed),
# (b) against the finest level k=47 (Nx=800).

def R2(P):
    a, b, c = P[1600], P[3200], P[6400]
    r1a = (4 * b - a) / 3; r1b = (4 * c - b) / 3
    return (8 * r1b - r1a) / 7


def load(path, eps, Nv=61):
    res = json.load(open(path))
    tab = {}
    for r in res:
        j = r['job']
        if j['eps'] != eps or j['Nv'] != Nv:
            continue
        tab.setdefault(j['k'], {})[j['Nt']] = r['prices']
    out = {}
    for k, d in tab.items():
        if all(n in d for n in (1600, 3200, 6400)):
            out[k] = {K: R2({n: d[n][K] for n in d}) for K in d[1600]}
    return out


def p_from_three(k1, k2, k3, d12, d23):
    """d12 = P(k2)-P(k1), d23 = P(k3)-P(k2); with e(k)=C k^-p, solve the ratio equation for p."""
    f = lambda p: (k1**-p - k2**-p) / (k2**-p - k3**-p) - d12 / d23
    try:
        return brentq(f, 0.2, 20.0)
    except ValueError:
        return float('nan')


def report(path, eps, ref_k=47, Nv=61):
    V = load(path, eps, Nv)
    ks = sorted(V)
    print(f"\n=== eps={eps}  Nv={Nv}  levels k={ks}")
    for K in ['90.0', '100.0', '110.0']:
        print(f" K={K}")
        vals = [V[k][K] for k in ks]
        for k, v in zip(ks, vals):
            print(f"   k={k:3d} Nx={17*k+1:4d}  P={v:.11f}")
        # (a) successive differences
        dd = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        ps = [p_from_three(ks[i], ks[i + 1], ks[i + 2], dd[i], dd[i + 1]) for i in range(len(dd) - 1)]
        print("   differences:", ["%.2e" % x for x in dd])
        print("   order from triples (no reference):", ["%.2f" % p for p in ps])
        # (b) against the finest level
        if ref_k in V:
            ref = V[ref_k][K]
            e = [abs(V[k][K] - ref) for k in ks if k != ref_k]
            kk = [k for k in ks if k != ref_k]
            sl = [np.log(e[i] / e[i + 1]) / np.log(kk[i + 1] / kk[i]) for i in range(len(e) - 1)]
            print("   |P-P_ref|:", ["%.2e" % x for x in e])
            print("   slopes against the finest level:", ["%.2f" % x for x in sl])


if __name__ == '__main__':
    import os
    path = sys.argv[1] if len(sys.argv) > 1 else 'C24.json'
    if path == 'C24.json' and not os.path.exists(path):
        json.dump(json.load(open('C2.json')) + json.load(open('C4.json')), open(path, 'w'))
    for eps in [0.25, 0.0]:
        try:
            report(path, eps)
        except Exception as ex:
            print("eps", eps, "no data:", ex)
