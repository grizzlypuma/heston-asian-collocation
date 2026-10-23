"""Full-truncation Euler scheme (Lord, Koekkoek & van Dijk 2010) for (S, v) under Heston.
Returns the continuous arithmetic average A_T/T and the log-geometric average Y_T (trapezoidal rule)."""
import numpy as np


def ft_euler(n, nst, kappa, theta, xi, rho, v0, S0, r, T, seed, q=0.0):
    rng = np.random.default_rng(seed); dt = T/nst; half = n//2; sq = np.sqrt(1-rho*rho)
    v = np.full(n, v0); x = np.full(n, np.log(S0)); S = np.exp(x)
    IA = 0.5*S*dt; IG = 0.5*x*dt
    for step in range(1, nst+1):
        Z1 = rng.standard_normal(half); Z1 = np.concatenate([Z1, -Z1])
        Z2 = rng.standard_normal(half); Z2 = np.concatenate([Z2, -Z2])
        vp = np.maximum(v, 0.0); sv = np.sqrt(vp*dt)
        x += (r - q - 0.5*vp)*dt + sv*(rho*Z1 + sq*Z2)
        v += kappa*(theta - vp)*dt + xi*sv*Z1
        S = np.exp(x); w = 0.5 if step == nst else 1.0
        IA += w*S*dt; IG += w*x*dt
    return IA/T, IG/T
