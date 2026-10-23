import numpy as np
from scipy.interpolate import BSpline
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import norm
import time

# ============================================================
# Geometric twin: quintic B-spline collocation (tests the machinery).
# PDE in tau = T - t, a(tau) = tau/T:
#   u_tau = L(tau) u,
#   L u = a(r - v/2)u_z + kappa(theta-v)u_v
#         + 0.5 a^2 v u_zz + rho*xi*a*v u_zv + 0.5 xi^2 v u_vv - r u
# Initial (tau=0): u = max(e^z - K, 0);  price: u(tau=T, z=ln S0, v=v0).
# L(tau) = a^2*M2 + a*M1 + M0: three matrices assembled once, one LU per step.
# BCs: z_lo: u=0 | z_hi: u_zz - u_z = 0 (exact for A e^z + B)
#      v_hi: u_v = 0 | v=0: no BC (Fichera).
# Model parameters are module-level (base set of the paper).
# ============================================================
kappa, theta, xi_, rho, v0 = 1.0, 0.09, 1.0, -0.3, 0.09
S0, r, T = 100.0, 0.04, 1.0


def geo_asian_bs_exact(S0, K, r, T, sigma):
    m = np.log(S0) + (r - 0.5*sigma**2)*T/2
    s2 = sigma**2*T/3
    s = np.sqrt(s2)
    d1 = (m - np.log(K) + s2)/s
    d2 = d1 - s
    return np.exp(-r*T)*(np.exp(m + 0.5*s2)*norm.cdf(d1) - K*norm.cdf(d2))


def sinh_grid(lo, hi, star, N, alpha):
    z = np.linspace(0, 1, N)
    s_lo = np.arcsinh((lo-star)/alpha); s_hi = np.arcsinh((hi-star)/alpha)
    return star + alpha*np.sinh(s_lo + z*(s_hi-s_lo))


def spline_1d(nodes, degree=5, kink=None):
    """Open knot vector. If a kink is given it is inserted with multiplicity
       `degree`, which lets the basis represent a jump in the first derivative
       EXACTLY (otherwise the payoff kink reduces the order to 2)."""
    a, b = nodes[0], nodes[-1]
    interior = nodes[1:-1]
    if kink is not None and a < kink < b:
        h = np.min(np.diff(nodes))
        interior = interior[np.abs(interior - kink) > 0.25*h]
        interior = np.sort(np.concatenate([interior, [kink]*4]))  # multiplicity 4 = C^1: nearly exact kink, u_zz still exists
    t = np.concatenate([[a]*(degree+1), interior, [b]*(degree+1)])
    nb = len(t)-degree-1
    xi = np.array([t[i+1:i+degree+1].mean() for i in range(nb)])
    B0 = np.zeros((nb, nb)); B1 = np.zeros((nb, nb)); B2 = np.zeros((nb, nb))
    for j in range(nb):
        spl = BSpline(t, np.eye(nb)[j], degree)
        B0[:, j] = spl(xi); B1[:, j] = spl.derivative(1)(xi); B2[:, j] = spl.derivative(2)(xi)
    return xi, B0, B1, B2, t, nb


def basis_rows(t, degree, nb, x0, ders=(0,)):
    out = []
    for d in ders:
        if d == 0:
            out.append(np.array([BSpline(t, np.eye(nb)[j], degree)(x0) for j in range(nb)]))
        else:
            out.append(np.array([BSpline(t, np.eye(nb)[j], degree).derivative(d)(x0) for j in range(nb)]))
    return out


def solve_geo(K=100.0, Nz=41, Nv=21, Nt=100, half_width=3.0, v_hi=1.5,
              az=0.25, av=0.15, degree=5, n_rann=4, theta_s=0.5,
              kap=None, th=None, xi_p=None, rh=None, v0_p=None, use_kink=True, verbose=False):
    kap = kappa if kap is None else kap
    th = theta if th is None else th
    xip = xi_ if xi_p is None else xi_p
    rh = rho if rh is None else rh
    v0p = v0 if v0_p is None else v0_p

    zc = np.log(S0)
    zn = sinh_grid(zc-half_width, zc+half_width, np.log(K), Nz, az)
    vn = sinh_grid(0.0, v_hi, 0.0, Nv, av)
    xz, B0z, B1z, B2z, tz, nbz = spline_1d(zn, degree, kink=np.log(K) if use_kink else None)
    xv, B0v, B1v, B2v, tv, nbv = spline_1d(vn, degree)

    Zc, Vc = np.meshgrid(xz, xv, indexing='ij')
    Zf, Vf = Zc.flatten(), Vc.flatten()
    Np = nbz*nbv

    S0z, S1z, S2z = map(sp.csr_matrix, (B0z, B1z, B2z))
    S0v, S1v, S2v = map(sp.csr_matrix, (B0v, B1v, B2v))
    M = sp.kron(S0z, S0v, format='csr')
    Dzz = sp.kron(S2z, S0v, format='csr')
    Dvv = sp.kron(S0z, S2v, format='csr')
    Dzv = sp.kron(S1z, S1v, format='csr')
    Dz = sp.kron(S1z, S0v, format='csr')
    Dv = sp.kron(S0z, S1v, format='csr')
    Dg = lambda a: sp.diags(a)

    # L(tau) = a^2*M2 + a*M1 + M0
    M2 = (Dg(0.5*Vf) @ Dzz).tocsr()
    M1 = (Dg(r - 0.5*Vf) @ Dz + Dg(rh*xip*Vf) @ Dzv).tocsr()
    M0 = (Dg(kap*(th - Vf)) @ Dv + Dg(0.5*xip**2*Vf) @ Dvv - r*M).tocsr()

    # ---- boundary rows ----
    idx = lambda i, j: i*nbv + j
    bc_rows = []
    rz_lo, = basis_rows(tz, degree, nbz, zn[0], (0,))
    rz_hi1, rz_hi2 = basis_rows(tz, degree, nbz, zn[-1], (1, 2))
    rv_hi1, = basis_rows(tv, degree, nbv, vn[-1], (1,))

    Bc = sp.lil_matrix((Np, Np))
    for j in range(nbv):
        rv = B0v[j, :]
        Bc[idx(0, j), :] = np.kron(rz_lo, rv)             # u = 0
        Bc[idx(nbz-1, j), :] = np.kron(rz_hi2 - rz_hi1, rv)  # u_zz - u_z = 0
        bc_rows += [idx(0, j), idx(nbz-1, j)]
    for i in range(1, nbz-1):
        rz = B0z[i, :]
        Bc[idx(i, nbv-1), :] = np.kron(rz, rv_hi1)        # u_v = 0
        bc_rows.append(idx(i, nbv-1))
    Bc = Bc.tocsr()

    bc_mask = np.zeros(Np, dtype=bool); bc_mask[bc_rows] = True
    bc_target = np.zeros(Np)   # all BCs are homogeneous
    Imask = sp.diags((~bc_mask).astype(float))
    Bmask = sp.diags(bc_mask.astype(float))
    masked = lambda A: (Imask @ A + Bmask @ Bc).tocsc()

    # initial condition (tau=0)
    u0 = np.maximum(np.exp(Zf) - K, 0.0)
    A_fit = masked(M)
    b_fit = u0.copy(); b_fit[bc_mask] = 0.0
    c = spla.splu(A_fit).solve(b_fit)

    dt = T/Nt
    for n in range(Nt):
        tau_n = n*dt
        tau_np1 = (n+1)*dt
        a_n = tau_n/T
        a_np1 = tau_np1/T
        L_n = (a_n**2)*M2 + a_n*M1 + M0
        L_np1 = (a_np1**2)*M2 + a_np1*M1 + M0
        th_s = 1.0 if n < n_rann else theta_s
        A = masked(M - th_s*dt*L_np1)
        R = Imask @ (M + (1.0-th_s)*dt*L_n)
        b = R @ c
        b[bc_mask] = 0.0
        c = spla.splu(A).solve(b)

    rz0, = basis_rows(tz, degree, nbz, np.log(S0), (0,))
    rv0, = basis_rows(tv, degree, nbv, v0p, (0,))
    return float(np.kron(rz0, rv0) @ c)


if __name__ == "__main__":
    print("="*76)
    print("TEST A: xi=0, rho=0, v0=theta  ->  exact Black-Scholes geometric formula")
    print("="*76)
    sig = np.sqrt(v0)
    for K in [90., 100., 110.]:
        exact = geo_asian_bs_exact(S0, K, r, T, sig)
        t0 = time.time()
        p = solve_geo(K=K, Nz=41, Nv=21, Nt=100, xi_p=1e-12, rh=0.0)
        el = time.time()-t0
        print(f"  K={K:5.0f}: solver={p:12.7f}  exact={exact:12.7f}  "
              f"rel.error={abs(p-exact)/exact:.2e}   ({el:.1f}s)")
