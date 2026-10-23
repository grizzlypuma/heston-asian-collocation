import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.stats import norm

# ============================================================
# Lower bound of Fusai & Kyriakou (2016) for the continuously monitored
# arithmetic Asian call under Heston (their Thm 3(i), eqs (43), (46), (50)).
#
#   LB(lam) = e^{-rT} E[(A/T - K) 1{Y_T > lam}],   Y_T = (1/T) int ln S dt
#   Phi(u; d) = e^{-rT}/(iu+d) [ (1/T) int_0^T phi_t(u-id) dt - K phi_0(u-id) ]
#   phi_t(y) = E[S_t e^{i y Y_T}],  phi_0(y) = E[e^{i y Y_T}]
#   LB(lam) = e^{-d lam}/pi int_0^inf Re[e^{-iu lam} Phi(u; d)] du,  MLB = max LB
#
# phi_t by two affine stages (iterated expectations, Appendix A):
#   ansatz exp(A + Bx X + Bv V) in tau (time to T), c = i y / T:
#     dBx/dtau = c   (+1 jump at tau = T - t)
#     dBv/dtau = Bx^2/2 - Bx/2 + rho*gamma*Bx*Bv + gamma^2 Bv^2/2 - alpha*Bv
#     dA/dtau  = (r-q) Bx + alpha*beta*Bv
# FK notation: alpha = kappa, beta = theta, gamma = xi.
# Computed with S0 = 1 (X0 = 0) and K~ = K/S0; the result is scaled by S0.
# ============================================================
def _rhs_factory(u_c, alpha, beta, gamma, rho, jump):
    """RHS for (Bv, Aint) in tau; Bx = c*tau + jump. u_c = c (vector)."""
    n = len(u_c)

    def rhs(tau, y):
        Bv = y[:n] + 1j * y[n:2 * n]
        Bx = u_c * tau + jump
        dBv = 0.5 * Bx * Bx - 0.5 * Bx + rho * gamma * Bx * Bv + 0.5 * gamma**2 * Bv * Bv - alpha * Bv
        dA = alpha * beta * Bv
        return np.concatenate([dBv.real, dBv.imag, dA.real, dA.imag])
    return rhs


def _integrate(rhs, t0, t1, y0, rtol, atol):
    if t1 <= t0:
        return y0.copy()
    sol = solve_ivp(rhs, (t0, t1), y0, method='DOP853', rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError("Riccati integration failed: " + sol.message)
    return sol.y[:, -1]


def phi_both(ups, t_nodes, T, r, q, V0, alpha, beta, gamma, rho, rtol=1e-12, atol=1e-14):
    """Returns (PHI, PHI_T):
       PHI[j, k]  = φ_{t_j,t_j,T}(-i/2, υ_k) = E[S_{t_j} e^{iυ_k Y_T}]  with S0=1
       PHI_T[k]   = φ_{T,T,T}(0, υ_k)       = E[e^{iυ_k Y_T}]          with S0=1."""
    ups = np.atleast_1d(np.asarray(ups, dtype=complex))
    n = len(ups)
    c = 1j * ups / T
    rhs1 = _rhs_factory(c, alpha, beta, gamma, rho, 0.0)
    rhs2 = _rhs_factory(c, alpha, beta, gamma, rho, 1.0)
    # stage 1, incrementally over the sorted tau0 = T - t_j
    tau0 = T - np.asarray(t_nodes, dtype=float)
    order = np.argsort(tau0)
    y = np.zeros(4 * n); tau_prev = 0.0
    stage1 = {}
    for j in order:
        y = _integrate(rhs1, tau_prev, tau0[j], y, rtol, atol)
        tau_prev = tau0[j]
        stage1[j] = y.copy()
    yT1 = _integrate(rhs1, tau_prev, T, y, rtol, atol)
    Bv = yT1[:n] + 1j * yT1[n:2 * n]; Ai = yT1[2 * n:3 * n] + 1j * yT1[3 * n:]
    # A = ∫(r-q)Bx dτ + ∫αβBv dτ ; without the jump ∫_0^T cτ dτ = cT²/2
    PHI_T = np.exp((r - q) * c * T * T / 2 + Ai + Bv * V0)
    PHI = np.empty((len(t_nodes), n), dtype=complex)
    for j in range(len(t_nodes)):
        yj = _integrate(rhs2, tau0[j], T, stage1[j], rtol, atol)
        Bv = yj[:n] + 1j * yj[n:2 * n]; Ai = yj[2 * n:3 * n] + 1j * yj[3 * n:]
        # with the jump: ∫(r-q)Bx = (r-q)(cT²/2 + (T - τ0_j)) = (r-q)(cT²/2 + t_j)
        PHI[j] = np.exp((r - q) * (c * T * T / 2 + t_nodes[j]) + Ai + Bv * V0)
    return PHI, PHI_T


class FKBound:
    """Fusai-Kyriakou lower bound for one parameter set; Phi(u) is computed once,
       after which LB(lambda) and MLB are cheap for any K."""

    def __init__(self, S0, r, q, T, V0, alpha, beta, gamma, rho,
                 delta=1.0, U=None, Nu=1024, Nt=48, rtol=1e-12, atol=1e-14):
        self.S0, self.r, self.q, self.T = S0, r, q, T
        self.delta = delta
        # scale in u: the standard deviation of Y_T is about sqrt(Vbar T/3)
        if U is None:
            sd = np.sqrt(max(V0, beta) * T / 3.0)
            U = 40.0 / sd
        self.U = U
        x, w = np.polynomial.legendre.leggauss(Nu)
        self.u = 0.5 * U * (x + 1.0); self.wu = 0.5 * U * w
        xt, wt = np.polynomial.legendre.leggauss(Nt)
        self.t = 0.5 * T * (xt + 1.0); self.wt = 0.5 * T * wt
        ups = self.u - 1j * delta
        PHI, PHI_T = phi_both(ups, self.t, T, r, q, V0, alpha, beta, gamma, rho, rtol, atol)
        self.I1 = (self.wt[:, None] * PHI).sum(axis=0) / T   # (1/T)∫ φ_{t,t,T} dt
        self.PHI_T = PHI_T
        self.pref = np.exp(-r * T) / (1j * self.u + delta)

    def Phi(self, Kt):
        return self.pref * (self.I1 - Kt * self.PHI_T)

    def LB(self, lam, K):
        """Lower bound for the parameter lambda (in absolute units of ln S)."""
        Kt = K / self.S0
        lt = lam - np.log(self.S0)
        Ph = self.Phi(Kt)
        val = np.exp(-self.delta * lt) / np.pi * np.sum(self.wu * np.real(np.exp(-1j * self.u * lt) * Ph))
        return self.S0 * val

    def MLB(self, K, width=0.5):
        lk = np.log(K)
        res = minimize_scalar(lambda l: -self.LB(l, K), bounds=(lk - width, lk + width),
                              method='bounded', options={'xatol': 1e-10})
        return -res.fun, res.x


# ---------- closed form for GBM (check in the limit gamma -> 0) ----------
def lb_gauss(lam, S0, K, r, q, sigma, T, Nt=200):
    X0 = np.log(S0); mu = r - q - 0.5 * sigma**2
    mY = X0 + mu * T / 2; sY = sigma * np.sqrt(T / 3.0)
    xt, wt = np.polynomial.legendre.leggauss(Nt)
    t = 0.5 * T * (xt + 1.0); w = 0.5 * T * wt
    cov = sigma**2 * (t - t * t / (2 * T))
    ES = np.exp(X0 + mu * t + 0.5 * sigma**2 * t)
    term = ES * norm.cdf((mY + cov - lam) / sY)
    return np.exp(-r * T) * (np.sum(w * term) / T - K * norm.cdf((mY - lam) / sY))


def mlb_gauss(S0, K, r, q, sigma, T):
    lk = np.log(K)
    res = minimize_scalar(lambda l: -lb_gauss(l, S0, K, r, q, sigma, T),
                          bounds=(lk - 0.5, lk + 0.5), method='bounded', options={'xatol': 1e-12})
    return -res.fun, res.x
