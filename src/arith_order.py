import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import json, os, sys, time
from arith_colloc import spline_1d, basis_rows, sinh_grid

# ============================================================
# Production solver used in the paper (Sections 3 and 5): same scheme as
# solve_arith_z0, with (1) an x-grid given by ONE fixed map for all levels
# (nl:nr = 6:11 nodes on the two sides of x = 0, N-1 = 17k), needed for
# clean order estimates, and (2) readout at several strikes from one solve.
# bc_v selects the condition at v_max: 'vv' (w_vv = 0, default) or 'v'.
# ============================================================
ML, MR = 6, 11


def grid_fixed_map(k, x_lo=-1.0, x_hi=5.0, ax=0.3):
    s_lo = np.arcsinh(x_lo / ax); s_hi = np.arcsinh(x_hi / ax)
    nl, nr = ML * k, MR * k
    s = np.concatenate([np.linspace(s_lo, 0.0, nl + 1), np.linspace(0.0, s_hi, nr + 1)[1:]])
    return ax * np.sinh(s)


def solve_multi(k, Nv=61, Nt=1600, x_lo=-1.0, x_hi=5.0, v_hi=1.5, ax=0.3, av=0.15,
                r=0.04, T=1.0, kappa=1.0, theta=0.09, xi_=1.0, rho=-0.3, v0=0.09,
                S0=100.0, Ks=(90.0, 100.0, 110.0), smooth_eps=0.0, n_rann=4, theta_s=0.5,
                degree=5, bc_v='vv'):
    xn = grid_fixed_map(k, x_lo, x_hi, ax)
    vn = sinh_grid(0.0, v_hi, 0.0, Nv, av)
    xc, B0x, B1x, B2x, tx, nbx = spline_1d(xn, degree, kink=0.0)
    vc, B0v, B1v, B2v, tv, nbv = spline_1d(vn, degree)
    Xc, Vc = np.meshgrid(xc, vc, indexing='ij')
    Xf, Vf = Xc.flatten(), Vc.flatten()
    Np = nbx * nbv
    S0x, S1x, S2x = map(sp.csr_matrix, (B0x, B1x, B2x))
    S0v, S1v, S2v = map(sp.csr_matrix, (B0v, B1v, B2v))
    M = sp.kron(S0x, S0v, format='csr')
    Dxx = sp.kron(S2x, S0v, format='csr'); Dvv = sp.kron(S0x, S2v, format='csr')
    Dxv = sp.kron(S1x, S1v, format='csr')
    Dx = sp.kron(S1x, S0v, format='csr'); Dv = sp.kron(S0x, S1v, format='csr')
    Dg = sp.diags
    L = (Dg(0.5 * Vf * Xf**2) @ Dxx - Dg(rho * xi_ * Vf * Xf) @ Dxv
         + Dg(0.5 * xi_**2 * Vf) @ Dvv - Dg(r * Xf + 1.0 / T) @ Dx
         + Dg(kappa * (theta - Vf) + rho * xi_ * Vf) @ Dv).tocsr()
    idx = lambda i, j: i * nbv + j
    rx_lo2, = basis_rows(tx, degree, nbx, xn[0], (2,))
    rx_hi0, = basis_rows(tx, degree, nbx, xn[-1], (0,))
    rv_hi2, = basis_rows(tv, degree, nbv, vn[-1], (2,) if bc_v == 'vv' else (1,))
    Bc = sp.lil_matrix((Np, Np)); bc_rows = []
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
    Imask = sp.diags((~bc_mask).astype(float)); Bmask = sp.diags(bc_mask.astype(float))
    masked = lambda A: (Imask @ A + Bmask @ Bc).tocsc()
    if smooth_eps > 0:
        w0 = 0.5 * (np.sqrt(Xf**2 + smooth_eps**2) - Xf)
    else:
        w0 = np.maximum(-Xf, 0.0)
    b_fit = w0.copy(); b_fit[bc_mask] = 0.0
    c = spla.splu(masked(M)).solve(b_fit)
    dt = T / Nt
    lu_r = spla.splu(masked(M - dt * L)); lu_c = spla.splu(masked(M - theta_s * dt * L))
    R_r = (Imask @ M).tocsr(); R_c = (Imask @ (M + (1.0 - theta_s) * dt * L)).tocsr()
    for n in range(Nt):
        if n < n_rann:
            b = R_r @ c; b[bc_mask] = 0.0; c = lu_r.solve(b)
        else:
            b = R_c @ c; b[bc_mask] = 0.0; c = lu_c.solve(b)
    rv0, = basis_rows(tv, degree, nbv, v0, (0,))
    out = {}
    for K in Ks:
        rx0, = basis_rows(tx, degree, nbx, K / S0, (0,))
        out[K] = S0 * float(np.kron(rx0, rv0) @ c)
    return out, len(xn)


def price_extrapolated(k=16, Nv=61, Nt=1600, **kw):
    """Prices {K: price} with two-level Richardson extrapolation in time over Nt, 2Nt, 4Nt
    (Section 3 of the paper). Other keyword arguments are passed to solve_multi."""
    P = [solve_multi(k, Nv=Nv, Nt=n, **kw)[0] for n in (Nt, 2 * Nt, 4 * Nt)]
    out = {}
    for K in P[0]:
        a, b, c = (p[K] for p in P)
        r1a, r1b = (4 * b - a) / 3, (4 * c - b) / 3
        out[K] = (8 * r1b - r1a) / 7
    return out


def run_jobs(jobs, path):
    """jobs: list of dict(k, Nv, Nt, eps, ...). Results are appended to the json; finished jobs are skipped."""
    res = json.load(open(path)) if os.path.exists(path) else []
    done = {json.dumps(r['job'], sort_keys=True) for r in res}
    for job in jobs:
        key = json.dumps(job, sort_keys=True)
        if key in done:
            continue
        t0 = time.time()
        kw = dict(job); eps = kw.pop('eps', 0.0)
        prices, Nx = solve_multi(smooth_eps=eps, **kw)
        rec = dict(job=job, Nx=Nx, prices={str(K): v for K, v in prices.items()}, sec=time.time() - t0)
        res.append(rec)
        json.dump(res, open(path, 'w'), indent=1)
        print(json.dumps(rec), flush=True)
    return res


if __name__ == '__main__':
    spec = json.load(open(sys.argv[1]))
    run_jobs(spec['jobs'], spec['out'])
