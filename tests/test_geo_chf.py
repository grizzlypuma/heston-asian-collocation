"""Exact geometric price: Black-Scholes limit and base-point values (Section 4.1)."""
import geo_chf as g


def test_bs_limit():
    for r, T, q in [(0.04, 1.0, 0.0), (0.1, 5.0, 0.02), (0.0, 0.25, 0.0)]:
        for K in (90., 100., 110.):
            ex = g.geo_asian_bs_exact(100., K, r, T, 0.3, q=q)
            p = g.price_carr_madan(K, xi=1e-10, rh=0.0, v0p=0.09, th=0.09, r=r, T=T, q=q)
            assert abs(p - ex) / ex < 1e-10


def test_base_values():
    ref = {90.: 13.3650818582, 100.: 6.7095975217, 110.: 2.7174933218}
    for K, v in ref.items():
        assert abs(g.price_carr_madan(K) - v) < 1e-9
