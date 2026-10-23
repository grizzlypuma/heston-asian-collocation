import numpy as np
from scipy.linalg import solve_banded
import time

# ============================================================
# Comparator for the geometric twin: modified Craig-Sneyd ADI on the same
# operator as geo_colloc.py (coefficients depend on time through a(tau)).
#   u_tau = L u,  L = a(r - v/2) d_z + kappa(theta-v) d_v + 0.5 a^2 v d_zz
#                     + rho*xi*a*v d_zv + 0.5 xi^2 v d_vv - r,   a = tau/T
# Splitting: L0 = mixed term, L1 = z part (-r/2), L2 = v part (-r/2).
# Model parameters are module-level (base set of the paper).
# ============================================================
kappa, theta, xi_, rho, v0 = 1.0, 0.09, 1.0, -0.3, 0.09
S0, r, T = 100.0, 0.04, 1.0


def sinh_grid(lo, hi, star, N, alpha):
    z = np.linspace(0, 1, N)
    s_lo = np.arcsinh((lo-star)/alpha); s_hi = np.arcsinh((hi-star)/alpha)
    return star + alpha*np.sinh(s_lo + z*(s_hi-s_lo))


def d1_d2(g):
    """Weights of the first and second derivative, 3 points, non-uniform grid (interior nodes)."""
    h = np.diff(g); hm = h[:-1]; hp = h[1:]
    a_m = -hp/(hm*(hm+hp)); a_0 = (hp-hm)/(hm*hp); a_p = hm/(hp*(hm+hp))
    b_m = 2.0/(hm*(hm+hp)); b_0 = -2.0/(hm*hp); b_p = 2.0/(hp*(hm+hp))
    return (a_m, a_0, a_p), (b_m, b_0, b_p)


def solve_adi(K=100.0, Nz=81, Nv=41, Nt=400, half_width=3.5, v_hi=6.0,
              az=0.25, av=0.4, theta_s=1.0/3.0, n_rann=4, smooth=False):
    zc = np.log(S0)
    z = sinh_grid(zc-half_width, zc+half_width, np.log(K), Nz, az)
    v = sinh_grid(0.0, v_hi, 0.0, Nv, av)
    (az_m, az_0, az_p), (bz_m, bz_0, bz_p) = d1_d2(z)
    (av_m, av_0, av_p), (bv_m, bv_0, bv_p) = d1_d2(v)
    iz = np.arange(1, Nz-1); iv = np.arange(1, Nv-1)
    dt = T/Nt
    # one-sided forward difference of second order at v=0 (non-uniform grid)
    h1 = v[1]-v[0]; h2 = v[2]-v[1]
    fwd2 = (-(2*h1+h2)/(h1*(h1+h2)), (h1+h2)/(h1*h2), -h1/(h2*(h1+h2)))

    # initial condition (tau=0)
    Z, V = np.meshgrid(z, v, indexing='ij')
    U = np.exp(Z) if smooth else np.maximum(np.exp(Z)-K, 0.0)

    def a_of(tau):
        return tau/T

    # ---------- operators ----------
    def L1(U, tau):
        """z-direction: a(r-v/2)u_z + 0.5 a^2 v u_zz - r/2 u."""
        a = a_of(tau)
        out = np.zeros_like(U)
        uz = az_m[:, None]*U[iz-1, :] + az_0[:, None]*U[iz, :] + az_p[:, None]*U[iz+1, :]
        uzz = bz_m[:, None]*U[iz-1, :] + bz_0[:, None]*U[iz, :] + bz_p[:, None]*U[iz+1, :]
        out[iz, :] = (a*(r - 0.5*v[None, :])*uz + 0.5*a**2*v[None, :]*uzz
                      - 0.5*r*U[iz, :])
        return out

    def L2(U, tau):
        """v-direction: kappa(theta-v)u_v + 0.5 xi^2 v u_vv - r/2 u.
           v=0: degenerate, drift points inward (Fichera) -> ONE-SIDED, SECOND ORDER."""
        out = np.zeros_like(U)
        uv = av_m[None, :]*U[:, iv-1] + av_0[None, :]*U[:, iv] + av_p[None, :]*U[:, iv+1]
        uvv = bv_m[None, :]*U[:, iv-1] + bv_0[None, :]*U[:, iv] + bv_p[None, :]*U[:, iv+1]
        out[:, iv] = ((kappa*(theta - v[iv]))[None, :]*uv
                      + 0.5*xi_**2*v[iv][None, :]*uvv - 0.5*r*U[:, iv])
        c0, c1, c2 = fwd2
        out[:, 0] = kappa*theta*(c0*U[:, 0] + c1*U[:, 1] + c2*U[:, 2]) - 0.5*r*U[:, 0]
        return out

    def L0(U, tau):
        """mixed term: rho*xi*a*v u_zv."""
        a = a_of(tau)
        out = np.zeros_like(U)
        uz = np.zeros_like(U)
        uz[iz, :] = az_m[:, None]*U[iz-1, :] + az_0[:, None]*U[iz, :] + az_p[:, None]*U[iz+1, :]
        uzv = np.zeros_like(U)
        uzv[:, iv] = av_m[None, :]*uz[:, iv-1] + av_0[None, :]*uz[:, iv] + av_p[None, :]*uz[:, iv+1]
        out[np.ix_(iz, iv)] = (rho*xi_*a*v[iv][None, :])*uzv[np.ix_(iz, iv)]
        return out

    def L(U, tau):
        return L0(U, tau) + L1(U, tau) + L2(U, tau)

    # ---------- implicit solvers ----------
    def solve1(RHS, tau, coef):
        """(I - coef*L1) Y = RHS in z. Band (2,1): at z_hi the condition u_zz-u_z=0
           (exact for A*e^z+B with ANY A; a hard Dirichlet condition would fix A=1)."""
        a = a_of(tau)
        Y = np.empty_like(RHS)
        A_, B_ = z[-1]-z[-2], z[-1]-z[-3]
        wa = (2*B_-B_**2) - (2*A_-A_**2)
        wb = -(2*B_-B_**2)
        wc = (2*A_-A_**2)
        for j in range(Nv):
            vj = v[j]
            lo = -coef*(a*(r-0.5*vj)*az_m + 0.5*a**2*vj*bz_m)
            di = 1.0 - coef*(a*(r-0.5*vj)*az_0 + 0.5*a**2*vj*bz_0 - 0.5*r)
            up = -coef*(a*(r-0.5*vj)*az_p + 0.5*a**2*vj*bz_p)
            ab = np.zeros((4, Nz))          # (l=2, u=1): ab[1+i-j, j]
            ab[1, :] = 1.0
            ab[1, 1:-1] = di
            ab[0, 2:] = up
            ab[2, :-2] = lo
            rhs = RHS[:, j].copy()
            # z_lo: Dirichlet u=0 (deep out of the money)
            ab[1, 0] = 1.0; rhs[0] = 0.0
            # z_hi: u_zz - u_z = 0
            ab[1, -1] = wa; ab[2, -2] = wb; ab[3, -3] = wc
            rhs[-1] = 0.0
            Y[:, j] = solve_banded((2, 1), ab, rhs)
        return Y

    def solve2(RHS, tau, coef):
        """(I - coef*L2) Y = RHS in v. Band (1,2): a third point is needed at v=0."""
        Y = np.empty_like(RHS)
        lo = -coef*(kappa*(theta-v[iv])*av_m + 0.5*xi_**2*v[iv]*bv_m)
        di = 1.0 - coef*(kappa*(theta-v[iv])*av_0 + 0.5*xi_**2*v[iv]*bv_0 - 0.5*r)
        up = -coef*(kappa*(theta-v[iv])*av_p + 0.5*xi_**2*v[iv]*bv_p)
        c0, c1, c2 = fwd2
        for i in range(Nz):
            ab = np.zeros((4, Nv))          # (l=1, u=2): ab[u+i-j, j]
            ab[2, :] = 1.0                  # main diagonal
            ab[2, 1:-1] = di
            ab[1, 2:] = up                  # first superdiagonal
            ab[3, :-2] = lo                 # subdiagonal
            # v=0: degenerate inward drift, second order (Fichera, no BC)
            ab[2, 0] = 1.0 - coef*(kappa*theta*c0 - 0.5*r)
            ab[1, 1] = -coef*kappa*theta*c1
            ab[0, 2] = -coef*kappa*theta*c2
            # v_hi: u_v = 0
            ab[2, -1] = 1.0; ab[3, -2] = -1.0
            rhs = RHS[i, :].copy()
            rhs[-1] = 0.0
            Y[i, :] = solve_banded((1, 2), ab, rhs)
        return Y

    def bc_vector(tau):
        """Dirichlet values on the z-boundaries at a given tau (for the smooth test
           the solution is ~ e^z * g; the asymptotics are taken from U itself, approximately)."""
        return None

    th = theta_s
    for n in range(Nt):
        tau_n = n*dt; tau_1 = (n+1)*dt
        LU = L(U, tau_n)
        Y0 = U + dt*LU
        if n < n_rann:
            # Rannacher: fully implicit step (Douglas with theta=1), damps the kink
            Y1 = solve1(Y0 - dt*L1(U, tau_n), tau_1, dt)
            U = solve2(Y1 - dt*L2(U, tau_n), tau_1, dt)
            continue
        Y = Y0.copy()
        Y1 = solve1(Y - th*dt*L1(U, tau_n), tau_1, th*dt)
        Y2 = solve2(Y1 - th*dt*L2(U, tau_n), tau_1, th*dt)
        # MCS correction of the mixed term
        Yh = Y0 + th*dt*(L0(Y2, tau_1) - L0(U, tau_n))
        Yt = Yh + (0.5 - th)*dt*(L(Y2, tau_1) - L(U, tau_n))
        Y1b = solve1(Yt - th*dt*L1(U, tau_n), tau_1, th*dt)
        Y2b = solve2(Y1b - th*dt*L2(U, tau_n), tau_1, th*dt)
        U = Y2b
        # BCs are already imposed inside solve1 (u=0 at z_lo, u_zz-u_z=0 at z_hi)

    # readout: bicubic interpolation
    from scipy.interpolate import RectBivariateSpline
    kx = min(3, Nz-1); ky = min(3, Nv-1)
    spl = RectBivariateSpline(z, v, U, kx=kx, ky=ky, s=0)
    return float(spl(np.log(S0), v0)[0, 0])


if __name__ == "__main__":
    ex_smooth = 97.346000202330
    print("="*76)
    print("ADI (MCS): test with the SMOOTH terminal condition u(T)=e^z")
    print(f"  exact = {ex_smooth:.10f}")
    print("="*76)
    prev = None
    for N in [21, 31, 41, 61]:
        t0 = time.time()
        p = solve_adi(Nz=N, Nv=max(21, N//2), Nt=800, smooth=True)
        e = abs(p - ex_smooth)
        o = "" if prev is None else f"  order={np.log(prev[1]/e)/np.log(N/prev[0]):5.2f}"
        print(f"  Nz={N:3d}: {p:.8f}  |err|={e:.3e}{o}  ({time.time()-t0:.0f}s)")
        prev = (N, e)
