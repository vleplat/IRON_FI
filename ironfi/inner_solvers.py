import time
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass
class IronfiInnerStats:
    n_newton: int
    final_residual_norm: float
    success: bool
    elapsed_s: float


def solve_resolvent_newton(
    *,
    w0: np.ndarray,
    center: np.ndarray,
    lam: float,
    grad: Callable[[np.ndarray], np.ndarray],
    hess: Callable[[np.ndarray], np.ndarray],
    tol: float = 1e-6,
    max_it: int = 20,
    backtracking: bool = True,
    beta: float = 0.5,
    bt_max: int = 10,
    use_cholesky: bool = True,
) -> Tuple[np.ndarray, IronfiInnerStats]:
    """
    Solve g(u)=u-center + lam*grad(u)=0 with Newton/LM; Jacobian J=I+lam*hess(u) (SPD).
    """
    t0 = time.time()
    d = w0.size
    u = w0.copy()

    def g(u_):
        return u_ - center + lam * grad(u_)

    success = False
    n_newton = 0
    res_norm = float(np.linalg.norm(g(u)))

    for k in range(max_it):
        n_newton = k + 1
        gu = g(u)
        res_norm = float(np.linalg.norm(gu))
        if res_norm <= tol:
            success = True
            break

        H = hess(u)
        J = np.eye(d) + lam * H

        rhs = -gu
        if use_cholesky:
            try:
                from scipy.linalg import cho_factor, cho_solve  # type: ignore

                cf = cho_factor(J, overwrite_a=False, check_finite=False)
                s = cho_solve(cf, rhs, check_finite=False)
            except Exception:
                s = np.linalg.solve(J, rhs)
        else:
            s = np.linalg.solve(J, rhs)

        if not backtracking:
            u = u + s
            continue

        base = res_norm if np.isfinite(res_norm) else np.inf
        step = 1.0
        accepted = False
        for _ in range(bt_max):
            u_try = u + step * s
            r_try = float(np.linalg.norm(g(u_try)))
            if np.isfinite(r_try) and r_try <= base:
                u = u_try
                accepted = True
                break
            step *= beta
        if not accepted:
            u = u + step * s

    elapsed = time.time() - t0
    stats = IronfiInnerStats(
        n_newton=n_newton,
        final_residual_norm=res_norm,
        success=success,
        elapsed_s=elapsed,
    )
    return u, stats

