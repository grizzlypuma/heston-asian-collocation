import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

# ============================================================
# Comparator: modified Craig-Sneyd ADI (in 't Hout & Foulon 2010) on the
# same reduced operator as the quintic collocation (Section 5.5).
#
# tau = T - t:   w_tau = L w,
#   L = 0.5 v x^2 d_xx - rho*xi*v*x d_xv + 0.5 xi^2 v d_vv
#       - (r x + 1/T) d_x + [kappa(theta-v) + rho*xi*v] d_v   (no reaction term)
# Initial condition (tau=0): w = max(-x, 0);  price = S0 * w(T, x0=K/S0, v0).
# Coefficients do not depend on tau, so every factorisation is built once.
# Splitting: L0 = mixed term, L1 = x part, L2 = v part.
# BCs: x_lo: w_xx=0 | x_hi: w=0 | v_hi: w_vv=0 | v=0: no BC (degenerate row).
# ============================================================
def sinh_grid(lo, hi, star, N, alpha):
    z = np.linspace(0, 1, N)
    s_lo = np.arcsinh((lo - star) / alpha)
    s_hi = np.arcsinh((hi - star) / alpha)
    return star + alpha * np.sinh(s_lo + z * (s_hi - s_lo))


def fd_matrices(g, fichera_at_left=False):
    """Sparse D1, D2 on a non-uniform grid (3-point, second order).
       Edges: one-sided second order. If fichera_at_left, node 0 uses a
       one-sided forward difference (degenerate drift pointing inward)."""
    N = len(g)
    D1 = sp.lil_matrix((N, N)); D2 = sp.lil_matrix((N, N))
    h = np.diff(g)
    for i in range(1, N - 1):
        hm, hp = h[i - 1], h[i]
        D1[i, i - 1] = -hp / (hm * (hm + hp))
        D1[i, i]     = (hp - hm) / (hm * hp)
        D1[i, i + 1] = hm / (hp * (hm + hp))
        D2[i, i - 1] = 2.0 / (hm * (hm + hp))
        D2[i, i]     = -2.0 / (hm * hp)
        D2[i, i + 1] = 2.0 / (hp * (hm + hp))
    # left edge: one-sided forward difference, second order
    h1, h2 = g[1] - g[0], g[2] - g[1]
    D1[0, 0] = -(2 * h1 + h2) / (h1 * (h1 + h2))
    D1[0, 1] = (h1 + h2) / (h1 * h2)
    D1[0, 2] = -h1 / (h2 * (h1 + h2))
    # right edge: one-sided backward difference, second order
    k1, k2 = g[-1] - g[-2], g[-2] - g[-3]
    D1[-1, -1] = (2 * k1 + k2) / (k1 * (k1 + k2))
    D1[-1, -2] = -(k1 + k2) / (k1 * k2)
    D1[-1, -3] = k1 / (k2 * (k1 + k2))
    return D1.tocsr(), D2.tocsr()


def d2_row_onesided(g, at_left=True):
    """One-sided 3-point second-derivative row (for the BCs w_xx=0, w_vv=0)."""
    if at_left:
        h1, h2 = g[1] - g[0], g[2] - g[1]
        return [(0, 2.0 / (h1 * (h1 + h2))), (1, -2.0 / (h1 * h2)),
                (2, 2.0 / (h2 * (h1 + h2)))]
    else:
        k1, k2 = g[-1] - g[-2], g[-2] - g[-3]
        return [(-1, 2.0 / (k1 * (k1 + k2))), (-2, -2.0 / (k1 * k2)),
                (-3, 2.0 / (k2 * (k1 + k2)))]


def solve_arith_adi(K=100.0, S0=100.0, r=0.04, T=1.0,
                    kappa=1.0, theta=0.09, xi_=1.0, rho=-0.3, v0=0.09,
                    Nx=161, Nv=41, Nt=400,
                    x_lo=-1.0, x_hi=5.0, v_hi=1.5,
                    ax=0.3, av=0.15, theta_s=1.0 / 3.0, n_rann=4, Ks=None):
    xg = sinh_grid(x_lo, x_hi, 0.0, Nx, ax)
    vg = sinh_grid(0.0, v_hi, 0.0, Nv, av)
    D1x, D2x = fd_matrices(xg)
    D1v, D2v = fd_matrices(vg)
    Ix, Iv = sp.identity(Nx, format='csr'), sp.identity(Nv, format='csr')

    X, V = np.meshgrid(xg, vg, indexing='ij')
    Xf, Vf = X.flatten(), V.flatten()
    Np = Nx * Nv
    Dg = lambda a: sp.diags(a)

    Dxx = sp.kron(D2x, Iv, format='csr')
    Dx  = sp.kron(D1x, Iv, format='csr')
    Dvv = sp.kron(Ix, D2v, format='csr')
    Dv  = sp.kron(Ix, D1v, format='csr')
    Dxv = sp.kron(D1x, D1v, format='csr')

    L1 = (Dg(0.5 * Vf * Xf**2) @ Dxx + Dg(-(r * Xf + 1.0 / T)) @ Dx).tocsr()
    L2 = (Dg(0.5 * xi_**2 * Vf) @ Dvv
          + Dg(kappa * (theta - Vf) + rho * xi_ * Vf) @ Dv).tocsr()
    L0 = (Dg(-rho * xi_ * Vf * Xf) @ Dxv).tocsr()

    # ---- boundary rows ----
    idx = lambda i, j: i * Nv + j
    rows_x, rows_v = [], []
    Bx = sp.lil_matrix((Np, Np))   # x-boundary conditions (enter solve1)
    Bv = sp.lil_matrix((Np, Np))   # v-boundary conditions (enter solve2)
    xlo_st = d2_row_onesided(xg, at_left=True)
    vhi_st = d2_row_onesided(vg, at_left=False)
    for j in range(Nv):
        for (off, val) in xlo_st:          # x_lo: w_xx = 0
            Bx[idx(0, j), idx(off, j)] = val
        Bx[idx(Nx - 1, j), idx(Nx - 1, j)] = 1.0   # x_hi: w = 0
        rows_x += [idx(0, j), idx(Nx - 1, j)]
    for i in range(1, Nx - 1):
        for (off, val) in vhi_st:          # v_hi: w_vv = 0
            Bv[idx(i, Nv - 1), idx(i, Nv + off if off < 0 else off)] = val
        rows_v.append(idx(i, Nv - 1))
    Bx, Bv = Bx.tocsr(), Bv.tocsr()

    bc = np.zeros(Np, dtype=bool); bc[rows_x] = True; bc[rows_v] = True
    mx = np.zeros(Np, dtype=bool); mx[rows_x] = True
    mv = np.zeros(Np, dtype=bool); mv[rows_v] = True
    Iin = sp.diags((~bc).astype(float))
    Px, Pv = sp.diags(mx.astype(float)), sp.diags(mv.astype(float))

    # the operators do not act on boundary rows
    L0 = (Iin @ L0).tocsr(); L1 = (Iin @ L1).tocsr(); L2 = (Iin @ L2).tocsr()
    L = (L0 + L1 + L2).tocsr()
    Msys = sp.identity(Np, format='csr')

    def build(coef):
        A = Iin @ (Msys - coef * L1) + Px @ Bx + Pv @ Bv
        B = Iin @ (Msys - coef * L2) + Px @ Bx + Pv @ Bv
        return spla.splu(A.tocsc()), spla.splu(B.tocsc())

    dt = T / Nt
    lu1_r, lu2_r = build(dt)              # Rannacher (BE)
    lu1_c, lu2_c = build(theta_s * dt)    # MCS
    # ^ ONE factorisation each: the coefficients do not depend on time

    U = np.maximum(-Xf, 0.0)
    U[bc] = 0.0

    def s1(rhs, lu):
        b = Iin @ rhs; b[bc] = 0.0
        return lu.solve(b)

    for n in range(Nt):
        LU_ = L @ U
        Y0 = U + dt * LU_
        if n < n_rann:
            Y1 = s1(Y0 - dt * (L1 @ U), lu1_r)
            U = s1(Y1 - dt * (L2 @ U), lu2_r)
            continue
        th = theta_s
        Y1 = s1(Y0 - th * dt * (L1 @ U), lu1_c)
        Y2 = s1(Y1 - th * dt * (L2 @ U), lu2_c)
        Yh = Y0 + th * dt * (L0 @ Y2 - L0 @ U)
        Yt = Yh + (0.5 - th) * dt * (L @ Y2 - LU_)
        Y1b = s1(Yt - th * dt * (L1 @ U), lu1_c)
        U = s1(Y1b - th * dt * (L2 @ U), lu2_c)

    from scipy.interpolate import RectBivariateSpline
    W = U.reshape(Nx, Nv)
    spl = RectBivariateSpline(xg, vg, W, kx=min(3, Nx - 1), ky=min(3, Nv - 1), s=0)
    if Ks is not None:
        return {Kk: S0 * float(spl(Kk / S0, v0)[0, 0]) for Kk in Ks}
    return S0 * float(spl(K / S0, v0)[0, 0])

