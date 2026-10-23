import numpy as np
from geo_chf import price_carr_madan

# ============================================================
# Monte Carlo check for arbitrary parameters: Andersen's QE scheme with the
# exact geometric Asian price (geo_chf) as control variate (Kemna-Vorst).
# geo_chf takes T, r, S0, q as arguments, so the control is exactly
# centred for any parameter set.
# ============================================================
def qe_paths(n, nst, kappa, theta, xi_, rho, v0, S0, r, T, seed, q=0.0):
    rng = np.random.default_rng(seed)
    dt = T / nst
    E = np.exp(-kappa * dt)
    half = n // 2
    v = np.full(n, v0)
    logS = np.full(n, np.log(S0))
    S = np.exp(logS)
    IA = 0.5 * S * dt
    IG = 0.5 * logS * dt
    K0 = -rho * kappa * theta / xi_ * dt
    K1 = 0.5 * dt * (kappa * rho / xi_ - 0.5) - rho / xi_
    K2 = 0.5 * dt * (kappa * rho / xi_ - 0.5) + rho / xi_
    K3 = 0.5 * dt * (1 - rho * rho); K4 = K3
    for step in range(1, nst + 1):
        m = theta + (v - theta) * E
        s2 = (v * xi_**2 * E / kappa) * (1 - E) + (theta * xi_**2 / (2 * kappa)) * (1 - E)**2
        psi = s2 / (m * m)
        vnew = np.empty_like(v)
        Zv = rng.standard_normal(half); Zv = np.concatenate([Zv, -Zv])
        U = rng.random(half); U = np.concatenate([U, U])
        b1 = psi <= 1.5
        ip = 2.0 / psi[b1]
        b = np.sqrt(ip - 1 + np.sqrt(ip * (ip - 1))); a_ = m[b1] / (1 + b * b)
        vnew[b1] = a_ * (b + Zv[b1])**2
        b2 = ~b1
        p = (psi[b2] - 1) / (psi[b2] + 1); beta = (1 - p) / m[b2]; u = U[b2]
        vnew[b2] = np.where(u <= p, 0.0, np.log((1 - p) / (1 - u)) / beta)
        Zx = rng.standard_normal(half); Zx = np.concatenate([Zx, -Zx])
        logS += (r - q) * dt + K0 + K1 * v + K2 * vnew + np.sqrt(np.maximum(K3 * v + K4 * vnew, 0)) * Zx
        v = vnew; S = np.exp(logS)
        w = 0.5 if step == nst else 1.0
        IA += w * S * dt
        IG += w * logS * dt
    return IA / T, np.exp(IG / T)


def mc_price(K=100.0, S0=100.0, r=0.04, T=1.0, kappa=1.0, theta=0.09,
             xi_=1.0, rho=-0.3, v0=0.09, n_paths=500_000, n_steps=256,
             seed=12345, batch=250_000, q=0.0, use_cv=True, diag=False):
    """Returns (price, s.e., correlation with the control, variance reduction factor).
    diag=True adds a dict: raw mean/s.e., mean of the control and its s.e.
    (centring check: E[c]=0 up to the discretisation bias)."""
    geo_exact = (price_carr_madan(K, kap=kappa, th=theta, xi=xi_, rh=rho, v0p=v0,
                                  S0=S0, r=r, T=T, q=q) if use_cv else 0.0)
    disc = np.exp(-r * T)
    Sy = Syy = Sc = Scc = Scy = 0.0
    n = 0
    done = 0; b = 0
    while done < n_paths:
        npb = min(batch, n_paths - done)
        A, G = qe_paths(npb, n_steps, kappa, theta, xi_, rho, v0, S0, r, T, seed + b, q=q)
        y = disc * np.maximum(A - K, 0.0)
        c = disc * np.maximum(G - K, 0.0) - geo_exact
        Sy += y.sum(); Syy += (y * y).sum()
        Sc += c.sum(); Scc += (c * c).sum(); Scy += (c * y).sum()
        n += npb; done += npb; b += 1
    raw_mean = Sy / n
    raw_var = Syy / n - raw_mean**2
    c_mean = Sc / n
    c_var = Scc / n - c_mean**2
    # coefficient from the centred moments
    cov = Scy / n - raw_mean * c_mean
    beta = (cov / c_var if (c_var > 0 and use_cv) else 0.0)
    mean = raw_mean - beta * c_mean
    var = raw_var - 2 * beta * cov + beta**2 * c_var
    corr = cov / np.sqrt(max(raw_var, 1e-300) * max(c_var, 1e-300))
    se = np.sqrt(max(var, 0) / n)
    raw_se = np.sqrt(max(raw_var, 0) / n)
    out = (mean, se, corr, (raw_se / se if se > 0 else 0.0))
    if diag:
        return out + ({'raw_mean': raw_mean, 'raw_se': raw_se, 'c_mean': c_mean,
                       'c_se': np.sqrt(max(c_var, 0) / n), 'beta': beta, 'geo_exact': geo_exact},)
    return out
