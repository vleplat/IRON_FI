import numpy as np

def ironfi_params(alpha: float, mu: float, gamma: float):
    """
    Compute IRON-FI parameters:
      tau  = 1/alpha + mu/gamma
      lam  = alpha / (gamma * (1 + tau))
    """
    tau = (1.0 / alpha) + (mu / gamma)
    lam = alpha / (gamma * (1.0 + tau))
    return tau, lam

def _solve_resolvent_quadratic(A, b, c, xi, lam):
    """
    Solve (I + lam A) x = c + xi + lam b
    Inputs:
      A: (n,n) SPD
      b: (n,)
      c: (n, m)  center per-sample (m samples)
      xi: (n, m) noise per-sample
      lam: scalar >= 0
    Returns:
      x: (n, m)
    """
    n = A.shape[0]
    M = np.eye(n) + lam * A
    rhs = c + xi + lam * b.reshape(-1, 1)
    # Use NumPy's dense solve for robustness (avoids occasional SciPy BLAS/LAPACK segfaults in some environments).
    x = np.linalg.solve(M, rhs)
    return x

def ironfi_quadratic_step(A, b, x, v, alpha, mu, gamma, xi):
    """
    One IRON-FI step for quadratic f(x)=0.5 x^T A x - b^T x.
    Inputs:
      A: (n,n) SPD
      b: (n,)
      x: (n,m)  current positions for m Monte Carlo samples
      v: (n,m)  current velocities
      alpha, mu, gamma: scalars
      xi: (n,m) noise center perturbation
    Returns:
      x_next, v_next, gamma_next, info
    """
    tau, lam = ironfi_params(alpha, mu, gamma)
    c = (v + tau * x) / (1.0 + tau)
    x_next = _solve_resolvent_quadratic(A, b, c, xi, lam)
    v_next = (x_next - x) / alpha + x_next
    gamma_next = (gamma + alpha * mu) / (1.0 + alpha)
    info = {"tau": tau, "lam": lam}
    return x_next, v_next, gamma_next, info
