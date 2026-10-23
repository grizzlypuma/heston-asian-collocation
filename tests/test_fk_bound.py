"""Lower bound of Fusai and Kyriakou under Heston (Section 4.2, Appendix A)."""
import numpy as np
import geo_chf as g
from fk_bound import FKBound, phi_both, mlb_gauss

BASE = dict(T=1., r=0.04, q=0., V0=0.09, alpha=1., beta=0.09, gamma=1.0, rho=-0.3)


def test_upsilon_zero_gives_forward():
    t = np.array([0.0, 0.3, 0.7, 1.0])
    PHI, PHIT = phi_both(np.array([0j]), t, **BASE)
    assert np.max(np.abs(PHI[:, 0] - np.exp(0.04 * t))) < 1e-13
    assert abs(PHIT[0] - 1) < 1e-13


def test_matches_geometric_chf():
    ups = np.linspace(0, 40, 17) - 1.0j
    _, PHIT = phi_both(ups, np.array([0.0]), **BASE)
    ref = g.chf(ups, kap=1., th=0.09, xi=1.0, rh=-0.3, v0p=0.09, S0=1.0, r=0.04, T=1.0)
    assert np.max(np.abs(PHIT - ref)) < 1e-12


def test_fk_table10_gaussian_limit():
    cases = [(2, 2, 0.02, 0.1, 1, 0.055985), (2, 2, 0.05, 0.5, 1, 0.246298), (2.1, 2, 0.05, 0.5, 1, 0.306094)]
    for S0, K, r, s, T, tab in cases:
        fb = FKBound(S0, r, 0.0, T, s * s, 1.0, s * s, 1e-9, 0.0)
        m, _ = fb.MLB(K)
        assert abs(m - tab) <= 5e-7
        assert abs(m - mlb_gauss(S0, K, r, 0.0, s, T)[0]) < 1e-10


def test_base_bound_values():
    fb = FKBound(S0=100., **BASE)
    ref = {90.: 13.755959, 100.: 7.028864, 110.: 2.963683}
    for K, v in ref.items():
        assert abs(fb.MLB(K)[0] - v) < 1e-6
