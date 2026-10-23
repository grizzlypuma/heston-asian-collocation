import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm

# ============================================================
# Exact price of the geometric Asian call under Heston (Section 4.1):
# joint characteristic function via a Riccati equation with time-dependent
# coefficients, then Carr-Madan inversion.  T, r, S0, q are arguments;
# the defaults are the base parameter set of the paper.
#
# zeta_T = (1/T) int_0^T ln S dt
#        = Lambda + int_0^T c(t) v_t dt + N(0, (1-rho^2) I_a2v)   (given the v-path)
#   Lambda = ln S0 + (r-q)T/2 - (rho/xi)(v0 + kappa*theta*T/2)
#   -> phi(u) = exp(iu*Lambda + A + B*v0)
# Riccati in tau = T - t, a = tau/T:
#   dB/dtau = 0.5*xi^2*B^2 - kappa*B + c(tau),  B(0) = 0
#   dA/dtau = kappa*theta*B,                    A(0) = 0
# Discount e^{-rT}; q enters the drift only.
# ============================================================
# base parameter set; used only as default values
kappa, theta, xi_, rho, v0 = 1.0, 0.09, 1.0, -0.3, 0.09
S0, r, T = 100.0, 0.04, 1.0
q = 0.0


def c_of_tau(tau, u, kap, th, xi, rh, T):
    """Coefficient of v_t in the exponent. u is a complex vector."""
    a = tau / T
    if xi > 0:
        lin = -0.5 * a + (rh / xi) * (1.0 / T + kap * a)
    else:
        lin = -0.5 * a
    return 1j * u * lin - 0.5 * u**2 * (1.0 - rh**2) * a**2


def riccati(u, kap=kappa, th=theta, xi=xi_, rh=rho, T=T, rtol=1e-12, atol=1e-14):
    """Solves the Riccati system, vectorised over u, on [0, T]. Returns A(T), B(T)."""
    u = np.atleast_1d(np.asarray(u, dtype=complex))
    n = len(u)

    def rhs(tau, y):
        B = y[2 * n:3 * n] + 1j * y[3 * n:]
        c = c_of_tau(tau, u, kap, th, xi, rh, T)
        dB = 0.5 * xi**2 * B**2 - kap * B + c
        dA = kap * th * B
        return np.concatenate([dA.real, dA.imag, dB.real, dB.imag])

    sol = solve_ivp(rhs, (0.0, T), np.zeros(4 * n), method='DOP853',
                    rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError("Riccati integration failed: " + sol.message)
    yT = sol.y[:, -1]
    return yT[:n] + 1j * yT[n:2 * n], yT[2 * n:3 * n] + 1j * yT[3 * n:]


def chf(u, kap=kappa, th=theta, xi=xi_, rh=rho, v0p=v0,
        S0=S0, r=r, T=T, q=q, rtol=1e-12, atol=1e-14):
    """ChF of zeta_T = (1/T) int ln S dt  (= Y_T in the notation of Fusai and Kyriakou)."""
    u = np.atleast_1d(np.asarray(u, dtype=complex))
    if xi > 0:
        Lam = np.log(S0) + (r - q) * T / 2 - (rh / xi) * (v0p + kap * th * T / 2)
    else:
        Lam = np.log(S0) + (r - q) * T / 2
    A, B = riccati(u, kap, th, xi, rh, T, rtol=rtol, atol=atol)
    return np.exp(1j * u * Lam + A + B * v0p)


def price_carr_madan(K, alpha=1.5, U=200.0, N=512,
                     kap=kappa, th=theta, xi=xi_, rh=rho, v0p=v0,
                     S0=S0, r=r, T=T, q=q, rtol=1e-12, atol=1e-14):
    """Price of the geometric Asian call (continuous average) by Carr-Madan."""
    k = np.log(K)
    x, w = np.polynomial.legendre.leggauss(N)
    vq = 0.5 * U * (x + 1.0); wq = 0.5 * U * w
    u = vq - (alpha + 1.0) * 1j
    ph = chf(u, kap, th, xi, rh, v0p, S0, r, T, q, rtol=rtol, atol=atol)
    denom = alpha**2 + alpha - vq**2 + 1j * (2 * alpha + 1.0) * vq
    psi = np.exp(-r * T) * ph / denom
    integ = np.real(np.exp(-1j * vq * k) * psi)
    return np.exp(-alpha * k) / np.pi * np.sum(wq * integ)


def geo_asian_bs_exact(S0, K, r, T, sigma, q=0.0):
    """Continuously monitored geometric Asian call under GBM (with dividend yield q)."""
    m = np.log(S0) + (r - q - 0.5 * sigma**2) * T / 2
    s2 = sigma**2 * T / 3
    s = np.sqrt(s2)
    d1 = (m - np.log(K) + s2) / s
    return np.exp(-r * T) * (np.exp(m + 0.5 * s2) * norm.cdf(d1) - K * norm.cdf(d1 - s))


if __name__ == "__main__":
    print("=" * 78)
    print("TEST 1: limit xi->0, rho=0, v0=theta -> exact Black-Scholes geometric price")
    print("=" * 78)
    sig = np.sqrt(v0)
    for K in [90., 100., 110.]:
        ex = geo_asian_bs_exact(S0, K, r, T, sig)
        p = price_carr_madan(K, xi=1e-10, rh=0.0)
        print(f"  K={K:5.0f}: ChF={p:14.10f}  exact={ex:14.10f}  rel={abs(p-ex)/ex:.2e}")

    print("\n" + "=" * 78)
    print("TEST 2: FULL Heston vs MC benchmark (16M paths)")
    print("=" * 78)
    bench = {90: (13.365014, 0.000153), 100: (6.709546, 0.000103), 110: (2.717492, 0.000150)}
    print(f"{'K':>5} {'ChF (exact)':>16} {'MC bench':>14} {'diff':>11} {'sigmas':>8}")
    for K in [90., 100., 110.]:
        p = price_carr_madan(K)
        b, se = bench[int(K)]
        print(f"{K:5.0f} {p:16.10f} {b:14.6f} {p-b:+11.6f} {(p-b)/se:+8.2f}")
