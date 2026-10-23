"""Exact geometric price against the Black-Scholes geometric formula in the limit xi -> 0 (Section 4.1)."""
import _setup  # noqa: F401
import geo_chf as g

worst = 0.0
for r in [0.0, 0.04, 0.1]:
    for T in [0.25, 1.0, 5.0]:
        for q in [0.0, 0.02]:
            errs = []
            for K in [90., 100., 110.]:
                ex = g.geo_asian_bs_exact(100., K, r, T, 0.3, q=q)
                p = g.price_carr_madan(K, xi=1e-10, rh=0.0, v0p=0.09, th=0.09, r=r, T=T, q=q)
                errs.append(abs(p - ex) / ex)
            worst = max(worst, max(errs))
            print(f"r={r:4.2f} T={T:4.2f} q={q:4.2f}: max rel. error {max(errs):.2e}")
print(f"worst: {worst:.2e}")
print("base-point prices:", {K: round(g.price_carr_madan(K), 10) for K in (90., 100., 110.)})
