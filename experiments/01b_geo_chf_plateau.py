"""Exact geometric price: stability in the damping parameter and quadrature (Section 4.1)."""
import _setup  # noqa: F401
import numpy as np, geo_chf as g
for T in [0.25, 3.0, 5.0]:
    print(f"T={T}")
    for K in [90., 100., 110.]:
        vals = {}
        for a in [0.5, 1.0, 1.5, 3.0]:
            vals[a] = g.price_carr_madan(K, alpha=a, T=T)
        vU = g.price_carr_madan(K, alpha=1.5, U=400., N=1024, T=T)
        spread = max(vals.values()) - min(vals.values())
        print(f"  K={K:5.0f}: a=1.5 {vals[1.5]:.10f}  spread over alpha {spread:.1e}  (U=400,N=1024) diff {vU-vals[1.5]:+.1e}  a=3 {vals[3.0]:.10f}")
