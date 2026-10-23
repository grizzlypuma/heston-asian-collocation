"""Greville collocation on -u'' + u = f, u = sin(pi x): order p - 1 in the maximum norm (Section 5.3)."""
import numpy as np
from scipy.interpolate import BSpline


def _solve(N, p):
    nodes = np.linspace(0, 1, N + 1)
    t = np.concatenate([[0.] * (p + 1), nodes[1:-1], [1.] * (p + 1)])
    nb = len(t) - p - 1
    xi = np.array([t[i + 1:i + p + 1].mean() for i in range(nb)])
    B0 = np.zeros((nb, nb)); B2 = np.zeros((nb, nb))
    for j in range(nb):
        s = BSpline(t, np.eye(nb)[j], p)
        B0[:, j] = s(xi); B2[:, j] = s.derivative(2)(xi)
    A = -B2 + B0; f = (np.pi**2 + 1) * np.sin(np.pi * xi)
    A[0] = B0[0]; f[0] = 0.0; A[-1] = B0[-1]; f[-1] = 0.0
    c = np.linalg.solve(A, f)
    xs = np.linspace(0, 1, 2001)
    return np.max(np.abs(BSpline(t, c, p)(xs) - np.sin(np.pi * xs)))


def test_orders():
    for p, expected in [(3, 2.0), (5, 4.0)]:
        e1, e2 = _solve(40, p), _solve(80, p)
        assert abs(np.log2(e1 / e2) - expected) < 0.05
