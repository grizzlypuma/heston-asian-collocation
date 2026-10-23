import numpy as np
from scipy.interpolate import BSpline
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

# ============================================================
# Arithmetic Asian call under Heston: quintic B-spline collocation.
# Similarity variable x = (K - A/T)/S, value V = S * w(t, x, v).
# PDE in time-to-maturity tau = T - t (coefficients independent of tau):
#   w_tau = 0.5 v x^2 w_xx - rho xi v x w_xv + 0.5 xi^2 v w_vv
#           - (r x + 1/T) w_x + [kappa(theta - v) + rho xi v] w_v
# Terminal: w = max(-x, 0) (kink at x = 0).  Price: C = S0 * w(0, K/S0, v0).
# BCs: x_hi: w=0 | x_lo: w_xx=0 | v_hi: w_vv=0 | v=0: none (Fichera).
# Two sparse LU factorisations (Rannacher BE step, Crank-Nicolson) serve the
# whole march. solve_arith_z0 uses a grid that contains x = 0 as a node.
# ============================================================
def sinh_grid(lo, hi, star, N, alpha):
    z = np.linspace(0, 1, N)
    s_lo = np.arcsinh((lo - star) / alpha)
    s_hi = np.arcsinh((hi - star) / alpha)
    return star + alpha * np.sinh(s_lo + z * (s_hi - s_lo))


def spline_1d(nodes, degree=5, kink=None):
    """Open knot vector; the kink is inserted with multiplicity 4 (C^1)."""
    a, b = nodes[0], nodes[-1]
    interior = nodes[1:-1]
    if kink is not None and a < kink < b:
        h = np.min(np.diff(nodes))
        interior = interior[np.abs(interior - kink) > 0.25 * h]
        interior = np.sort(np.concatenate([interior, [kink] * 4]))
    t = np.concatenate([[a] * (degree + 1), interior, [b] * (degree + 1)])
    nb = len(t) - degree - 1
    xi = np.array([t[i + 1:i + degree + 1].mean() for i in range(nb)])
    B0 = np.zeros((nb, nb)); B1 = np.zeros((nb, nb)); B2 = np.zeros((nb, nb))
    eye = np.eye(nb)
    for j in range(nb):
        spl = BSpline(t, eye[j], degree)
        B0[:, j] = spl(xi)
        B1[:, j] = spl.derivative(1)(xi)
        B2[:, j] = spl.derivative(2)(xi)
    return xi, B0, B1, B2, t, nb


def basis_rows(t, degree, nb, x0, ders=(0,)):
    out = []
    eye = np.eye(nb)
    for d in ders:
        if d == 0:
            out.append(np.array([BSpline(t, eye[j], degree)(x0) for j in range(nb)]))
        else:
            out.append(np.array([BSpline(t, eye[j], degree).derivative(d)(x0)
                                 for j in range(nb)]))
    return out


def solve_arith(K=100.0, S0=100.0, r=0.04, T=1.0,
                kappa=1.0, theta=0.09, xi_=1.0, rho=-0.3, v0=0.09,
                Nx=141, Nv=31, Nt=1600,
                x_lo=-1.0, x_hi=5.0, v_hi=4.0,
                ax=0.3, av=0.15, degree=5, n_rann=4, theta_s=0.5,
                verbose=False):
    # --- grids ---
    xn = sinh_grid(x_lo, x_hi, 0.0, Nx, ax)      # clustering at x=0 (kink and readout)
    vn = sinh_grid(0.0, v_hi, 0.0, Nv, av)       # clustering at v=0
    xc, B0x, B1x, B2x, tx, nbx = spline_1d(xn, degree, kink=0.0)
    vc, B0v, B1v, B2v, tv, nbv = spline_1d(vn, degree)

    Xc, Vc = np.meshgrid(xc, vc, indexing='ij')
    Xf, Vf = Xc.flatten(), Vc.flatten()
    Np = nbx * nbv

    S0x, S1x, S2x = map(sp.csr_matrix, (B0x, B1x, B2x))
    S0v, S1v, S2v = map(sp.csr_matrix, (B0v, B1v, B2v))
    M   = sp.kron(S0x, S0v, format='csr')
    Dxx = sp.kron(S2x, S0v, format='csr')
    Dvv = sp.kron(S0x, S2v, format='csr')
    Dxv = sp.kron(S1x, S1v, format='csr')
    Dx  = sp.kron(S1x, S0v, format='csr')
    Dv  = sp.kron(S0x, S1v, format='csr')
    Dg = lambda a: sp.diags(a)

    # L (backward time); the coefficients do not depend on time
    L = (Dg(0.5 * Vf * Xf**2) @ Dxx
         - Dg(rho * xi_ * Vf * Xf) @ Dxv
         + Dg(0.5 * xi_**2 * Vf) @ Dvv
         - Dg(r * Xf + 1.0 / T) @ Dx
         + Dg(kappa * (theta - Vf) + rho * xi_ * Vf) @ Dv).tocsr()

    # ---- boundary rows ----
    idx = lambda i, j: i * nbv + j
    rx_lo2, = basis_rows(tx, degree, nbx, xn[0], (2,))    # w_xx = 0
    rx_hi0, = basis_rows(tx, degree, nbx, xn[-1], (0,))   # w = 0
    rv_hi2, = basis_rows(tv, degree, nbv, vn[-1], (2,))   # w_vv = 0

    Bc = sp.lil_matrix((Np, Np))
    bc_rows = []
    for j in range(nbv):
        rv = B0v[j, :]
        Bc[idx(0, j), :] = np.kron(rx_lo2, rv)
        Bc[idx(nbx - 1, j), :] = np.kron(rx_hi0, rv)
        bc_rows += [idx(0, j), idx(nbx - 1, j)]
    for i in range(1, nbx - 1):
        rx = B0x[i, :]
        Bc[idx(i, nbv - 1), :] = np.kron(rx, rv_hi2)
        bc_rows.append(idx(i, nbv - 1))
    Bc = Bc.tocsr()

    bc_mask = np.zeros(Np, dtype=bool); bc_mask[bc_rows] = True
    Imask = sp.diags((~bc_mask).astype(float))
    Bmask = sp.diags(bc_mask.astype(float))
    masked = lambda A: (Imask @ A + Bmask @ Bc).tocsc()

    # --- initial condition (tau=0): w = max(-x, 0) ---
    w0 = np.maximum(-Xf, 0.0)
    b_fit = w0.copy(); b_fit[bc_mask] = 0.0
    c = spla.splu(masked(M)).solve(b_fit)

    # --- time march: one LU for the Rannacher steps and one for the main scheme ---
    dt = T / Nt
    A_ran = masked(M - 1.0 * dt * L)
    A_cn  = masked(M - theta_s * dt * L)
    lu_ran = spla.splu(A_ran)
    lu_cn  = spla.splu(A_cn)
    R_ran = Imask @ M
    R_cn  = Imask @ (M + (1.0 - theta_s) * dt * L)

    for n in range(Nt):
        if n < n_rann:
            b = R_ran @ c; b[bc_mask] = 0.0
            c = lu_ran.solve(b)
        else:
            b = R_cn @ c; b[bc_mask] = 0.0
            c = lu_cn.solve(b)

    x0 = K / S0
    rx0, = basis_rows(tx, degree, nbx, x0, (0,))
    rv0, = basis_rows(tv, degree, nbv, v0, (0,))
    return S0 * float(np.kron(rx0, rv0) @ c)


if __name__ == "__main__":
    print("=" * 72)
    print("RECONSTRUCTION CHECK: Corsaro set (Feller condition violated)")
    print("  early MC benchmark (geometric control variate): K90=13.760462  K100=7.032640"
          "  K110=2.967759  (+-0.0003)")
    print("=" * 72)
    bench = {90.0: 13.760462, 100.0: 7.032640, 110.0: 2.967759}
    for K in [90.0, 100.0, 110.0]:
        t0 = time.time()
        p = solve_arith(K=K, Nx=141, Nv=31, Nt=1600, v_hi=4.0)
        print(f"  K={K:5.0f}: solver={p:11.6f}  benchmark={bench[K]:11.6f}"
              f"  diff={p-bench[K]:+9.6f}   ({time.time()-t0:.0f}s)")


def sinh_grid_through_zero(lo, hi, N, alpha):
    """sinh grid that is GUARANTEED to contain x=0 as a node.
       Needed for order measurements: otherwise the number of nodes dropped near the kink
       jumps with N, the basis changes discontinuously and the observed slope is noise."""
    s_lo = np.arcsinh(lo / alpha)
    s_hi = np.arcsinh(hi / alpha)
    frac = -s_lo / (s_hi - s_lo)
    nl = max(2, int(round(frac * (N - 1))))
    nr = (N - 1) - nl
    s_left = np.linspace(s_lo, 0.0, nl + 1)
    s_right = np.linspace(0.0, s_hi, nr + 1)[1:]
    s = np.concatenate([s_left, s_right])
    return alpha * np.sinh(s)


def solve_arith_z0(K=100.0, S0=100.0, r=0.04, T=1.0,
                   kappa=1.0, theta=0.09, xi_=1.0, rho=-0.3, v0=0.09,
                   Nx=141, Nv=61, Nt=1600, x_lo=-1.0, x_hi=5.0, v_hi=1.5,
                   ax=0.3, av=0.15, degree=5, n_rann=4, theta_s=0.5,
                   smooth_eps=0.0):
    """Like solve_arith, but the x-grid passes exactly through x=0.
       smooth_eps>0: smoothed terminal condition 0.5*(sqrt(x^2+eps^2)-x), a test without the kink."""
    import scipy.sparse as _sp, scipy.sparse.linalg as _spla
    xn = sinh_grid_through_zero(x_lo, x_hi, Nx, ax)
    vn = sinh_grid(0.0, v_hi, 0.0, Nv, av)
    xc, B0x, B1x, B2x, tx, nbx = spline_1d(xn, degree, kink=0.0)
    vc, B0v, B1v, B2v, tv, nbv = spline_1d(vn, degree)
    Xc, Vc = np.meshgrid(xc, vc, indexing='ij')
    Xf, Vf = Xc.flatten(), Vc.flatten()
    Np = nbx * nbv
    S0x, S1x, S2x = map(_sp.csr_matrix, (B0x, B1x, B2x))
    S0v, S1v, S2v = map(_sp.csr_matrix, (B0v, B1v, B2v))
    M = _sp.kron(S0x, S0v, format='csr')
    Dxx = _sp.kron(S2x, S0v, format='csr'); Dvv = _sp.kron(S0x, S2v, format='csr')
    Dxv = _sp.kron(S1x, S1v, format='csr')
    Dx = _sp.kron(S1x, S0v, format='csr'); Dv = _sp.kron(S0x, S1v, format='csr')
    Dg = lambda a: _sp.diags(a)
    L = (Dg(0.5 * Vf * Xf**2) @ Dxx - Dg(rho * xi_ * Vf * Xf) @ Dxv
         + Dg(0.5 * xi_**2 * Vf) @ Dvv - Dg(r * Xf + 1.0 / T) @ Dx
         + Dg(kappa * (theta - Vf) + rho * xi_ * Vf) @ Dv).tocsr()
    idx = lambda i, j: i * nbv + j
    rx_lo2, = basis_rows(tx, degree, nbx, xn[0], (2,))
    rx_hi0, = basis_rows(tx, degree, nbx, xn[-1], (0,))
    rv_hi2, = basis_rows(tv, degree, nbv, vn[-1], (2,))
    Bc = _sp.lil_matrix((Np, Np)); bc_rows = []
    for j in range(nbv):
        rv = B0v[j, :]
        Bc[idx(0, j), :] = np.kron(rx_lo2, rv)
        Bc[idx(nbx - 1, j), :] = np.kron(rx_hi0, rv)
        bc_rows += [idx(0, j), idx(nbx - 1, j)]
    for i in range(1, nbx - 1):
        Bc[idx(i, nbv - 1), :] = np.kron(B0x[i, :], rv_hi2)
        bc_rows.append(idx(i, nbv - 1))
    Bc = Bc.tocsr()
    bc_mask = np.zeros(Np, dtype=bool); bc_mask[bc_rows] = True
    Imask = _sp.diags((~bc_mask).astype(float))
    Bmask = _sp.diags(bc_mask.astype(float))
    masked = lambda A: (Imask @ A + Bmask @ Bc).tocsc()
    if smooth_eps > 0:
        w0 = 0.5 * (np.sqrt(Xf**2 + smooth_eps**2) - Xf)
    else:
        w0 = np.maximum(-Xf, 0.0)
    b_fit = w0.copy(); b_fit[bc_mask] = 0.0
    c = _spla.splu(masked(M)).solve(b_fit)
    dt = T / Nt
    lu_r = _spla.splu(masked(M - dt * L)); lu_c = _spla.splu(masked(M - theta_s * dt * L))
    R_r = Imask @ M; R_c = Imask @ (M + (1.0 - theta_s) * dt * L)
    for n in range(Nt):
        if n < n_rann:
            b = R_r @ c; b[bc_mask] = 0.0; c = lu_r.solve(b)
        else:
            b = R_c @ c; b[bc_mask] = 0.0; c = lu_c.solve(b)
    rx0, = basis_rows(tx, degree, nbx, K / S0, (0,))
    rv0, = basis_rows(tv, degree, nbv, v0, (0,))
    return S0 * float(np.kron(rx0, rv0) @ c)
