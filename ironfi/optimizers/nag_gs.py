from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def nag_gs_step(
    *,
    x: np.ndarray,
    v: np.ndarray,
    gamma: float,
    alpha: float,
    mu: float,
    grad_at_xnext: Callable[[np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    NAG-GS update from Algorithm alg:nag_gsgeneral (Gauss–Seidel discretization).

    Parameters
    ----------
    x : np.ndarray
        Current iterate.
    v : np.ndarray
        Auxiliary state.
    gamma : float
        Current damping parameter.
    alpha : float
        Stepsize parameter.
    mu : float
        Damping / strong convexity parameter used by the method.
    grad_at_xnext : callable
        Gradient oracle evaluated at x_{k+1} (semi-implicit).

    Returns
    -------
    x_next : np.ndarray
        Updated iterate.
    v_next : np.ndarray
        Updated auxiliary state.
    gamma_next : float
        Updated damping parameter.

    Notes:
    - Uses gradient at x_{k+1} (semi-implicit).
    - gamma is updated AFTER (x_{k+1}, v_{k+1}), using implicit Euler step.
    - If mu=0, we use the continuous extension mu^{-1} b_k = alpha/gamma.
    """
    a = alpha / (1.0 + alpha)
    x_next = (1.0 - a) * x + a * v

    b = (alpha * mu) / (alpha * mu + gamma) if (alpha * mu + gamma) != 0 else 0.0
    g = grad_at_xnext(x_next)

    if mu == 0.0:
        muinv_b = alpha / gamma
    else:
        muinv_b = b / mu

    v_next = (1.0 - b) * v + b * x_next - muinv_b * g
    gamma_next = (1.0 - a) * gamma + a * mu
    return x_next, v_next, gamma_next

