from typing import Callable, Dict, Tuple

import numpy as np

from .inner_solvers import solve_resolvent_newton


def ironfi_params(alpha: float, mu: float, gamma: float) -> tuple[float, float]:
    tau = (1.0 / alpha) + (mu / gamma)
    lam = alpha / (gamma * (1.0 + tau))
    return tau, lam


def update_gamma(gamma: float, alpha: float, mu: float) -> float:
    return (gamma + alpha * mu) / (1.0 + alpha)


def ironfi_step(
    *,
    x: np.ndarray,
    v: np.ndarray,
    alpha: float,
    mu: float,
    gamma: float,
    sigma: float,
    rng: np.random.Generator,
    grad: Callable[[np.ndarray], np.ndarray],
    hess: Callable[[np.ndarray], np.ndarray],
    inner_tol: float,
    inner_max_it: int,
    update_gamma_flag: bool = True,
    inner_use_cholesky: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
    """
    Generic IRON-FI outer step in R^d with center-perturbation noise.
    """
    tau, lam = ironfi_params(alpha, mu, gamma)
    c = (v + tau * x) / (1.0 + tau)
    xi = (np.sqrt(alpha) / (1.0 + tau)) * sigma * rng.normal(size=x.shape)
    center = c + xi

    x_next, inner = solve_resolvent_newton(
        w0=x,
        center=center,
        lam=lam,
        grad=grad,
        hess=hess,
        tol=inner_tol,
        max_it=inner_max_it,
        backtracking=True,
        use_cholesky=inner_use_cholesky,
    )

    v_next = x_next + (x_next - x) / alpha
    gamma_next = update_gamma(gamma, alpha, mu) if update_gamma_flag else gamma
    info = {"tau": tau, "lam": lam, "inner": inner}
    return x_next, v_next, gamma_next, info

