"""Tables 4 and 5 of the paper, rebuilt from the stored convergence runs.

Table 5 (time convergence) comes from C1.json: k=16 (Nx=273), Nv=61, payoff, Nt=800..25600.
Table 4 (converged prices) uses k=94 (Nx=1599), Nv=61 from C4.json with two-level Richardson over
Nt=1600/3200/6400, plus a v-correction taken on a coarser grid (k=32, Nv=91 minus Nv=61, from C2.json),
and compares with the pooled reference in E8_pooled.json. Needs no new computation.
"""
import _setup  # noqa: F401
import json

Ks = ['90.0', '100.0', '110.0']


def by_nt(runs, k, Nv, eps):
    return {r['job']['Nt']: r['prices'] for r in runs
            if r['job']['k'] == k and r['job']['Nv'] == Nv and r['job']['eps'] == eps}


def r2(P, K, nts=(1600, 3200, 6400)):
    a, b, c = (P[n][K] for n in nts)
    r1a, r1b = (4 * b - a) / 3, (4 * c - b) / 3
    return (8 * r1b - r1a) / 7


# ---------- Table 5: time convergence at K=100 ----------
C1 = json.load(open('C1.json'))
P = by_nt(C1, 16, 61, 0.0)
nts = sorted(P)
print('Table 5: time convergence, K=100, Nx=273, Nv=61 (payoff)')
print(f"{'Nt':>6} {'price':>14} {'d':>11} {'ratio':>6} {'R1':>14} {'d_R':>10} {'ratio':>6}")
prev = prev_d = prev_r1 = prev_dr = None
for n in nts:
    p = P[n]['100.0']
    d = None if prev is None else p - prev
    r1 = None if prev is None else (4 * p - prev) / 3
    dr = None if (r1 is None or prev_r1 is None) else r1 - prev_r1
    f = lambda x, fmt: '' if x is None else format(x, fmt)
    ratio = '' if (d is None or prev_d is None) else f"{prev_d / d:.2f}"
    ratio_r = '' if (dr is None or prev_dr is None) else f"{prev_dr / dr:.2f}"
    print(f"{n:>6} {p:14.10f} {f(d, '11.2e'):>11} {ratio:>6} {f(r1, '14.10f'):>14} {f(dr, '10.1e'):>10} {ratio_r:>6}")
    prev, prev_d, prev_r1, prev_dr = p, d, r1, dr

# remaining time error of R2(1600,3200,6400) against R2(6400,12800,25600)
for eps in (0.0, 0.25):
    Q = by_nt(C1, 16, 61, eps)
    worst = max(abs(r2(Q, K) - r2(Q, K, (6400, 12800, 25600))) for K in Ks)
    print(f"eps={eps}: |R2(1600..6400) - R2(6400..25600)| max over K = {worst:.1e}")

# ---------- Table 4: converged prices against the reference ----------
C2 = json.load(open('C2.json'))
C4 = json.load(open('C4.json'))
fine = by_nt(C4, 94, 61, 0.0)
c61, c91 = by_nt(C2, 32, 61, 0.0), by_nt(C2, 32, 91, 0.0)
E = json.load(open('E8_pooled.json'))
print('\nTable 4: converged collocation against the reference (base parameters)')
for K in Ks:
    dv = r2(c91, K) - r2(c61, K)
    p = r2(fine, K) + dv
    ref, se = E[K]['E8'], E[K]['se']
    print(f"K={float(K):5.0f}: collocation {p:.7f} (v-correction {dv:+.1e}) | reference {ref:.6f} ± {se:.1e}"
          f" | diff {p - ref:+.1e} = {(p - ref) / se:+.2f} s.e.")
