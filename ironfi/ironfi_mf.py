from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


@dataclass
class CGStats:
    n_cg: int
    final_residual_norm: float
    success: bool


def _norm(x: np.ndarray) -> float:
    # Suppress occasional spurious FP warnings from BLAS/norm; check finiteness at call sites if needed.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        return float(np.linalg.norm(x))


def cg_solve(
    Ax: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    tol: float = 1e-6,
    max_it: int = 200,
) -> Tuple[np.ndarray, CGStats]:
    """
    Conjugate gradient for SPD systems Ax=b using only matvec.
    Stops when ||r|| <= tol * ||b|| (relative) or <= tol (absolute if ||b||=0).

    Parameters
    ----------
    Ax : callable
        Function implementing matrix-vector product v -> A v.
    b : np.ndarray
        Right-hand side vector.
    x0 : np.ndarray or None, default=None
        Optional initial guess.
    tol : float, default=1e-6
        Relative tolerance (w.r.t. ||b||) unless ||b||=0, then absolute.
    max_it : int, default=200
        Maximum number of CG iterations.

    Returns
    -------
    x : np.ndarray
        Approximate solution.
    stats : CGStats
        Iteration count and final residual norm.
    """
    x = np.zeros_like(b) if x0 is None else x0.copy()
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        r = b - Ax(x)
        bnorm = _norm(b)
    thresh = tol * bnorm if bnorm > 0 else tol
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        rsold = float(np.dot(r, r))
    rnorm = float(np.sqrt(rsold))
    if rnorm <= thresh:
        return x, CGStats(n_cg=0, final_residual_norm=rnorm, success=True)
    p = r.copy()
    n_cg = 0

    for k in range(max_it):
        Ap = Ax(p)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
            denom = float(np.dot(p, Ap))
        if denom <= 0:
            # not SPD / numerical issue
            break
        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * Ap
        with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
            rsnew = float(np.dot(r, r))
        n_cg = k + 1
        rnorm = float(np.sqrt(rsnew))
        if rnorm <= thresh:
            return x, CGStats(n_cg=n_cg, final_residual_norm=rnorm, success=True)
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x, CGStats(n_cg=n_cg, final_residual_norm=float(np.sqrt(rsold)), success=False)


@dataclass
class IronfiMFInnerStats:
    n_newton: int
    n_cg_total: int
    final_residual_norm: float
    success: bool
    elapsed_s: float


def solve_resolvent_newton_cg(
    *,
    u0: np.ndarray,
    center: np.ndarray,
    lam: float,
    grad: Callable[[np.ndarray], np.ndarray],
    hvp: Callable[[np.ndarray, np.ndarray], np.ndarray],
    tol: float = 1e-6,
    max_newton: int = 20,
    cg_tol: float = 1e-4,
    cg_max_it: int = 200,
    backtracking: bool = True,
    beta: float = 0.5,
    max_ls: int = 10,
) -> Tuple[np.ndarray, IronfiMFInnerStats]:
    """
    Solve g(u)=u-center + lam*grad(u)=0 with Newton steps.
    Each Newton step solves (I + lam*H(u)) s = -g(u) via CG using Hessian-vector products.

    hvp(u, v) must return H(u) v.

    Parameters
    ----------
    u0 : np.ndarray
        Initial guess for the inner solve.
    center : np.ndarray
        Prox center (c + xi).
    lam : float
        Resolvent/LM parameter.
    grad : callable
        Gradient function at u.
    hvp : callable
        Hessian-vector product function (u, v) -> H(u)v.
    tol : float
        Inner residual tolerance on ||g(u)||.
    max_newton : int
        Maximum Newton iterations.
    cg_tol : float
        Relative CG tolerance per Newton step.
    cg_max_it : int
        Maximum CG iterations per Newton step.

    Returns
    -------
    u : np.ndarray
        Approximate solution.
    stats : IronfiMFInnerStats
        Newton/CG counts and final residual.
    """
    t0 = time.time()
    u = u0.copy()
    n_cg_total = 0
    success = False
    res_norm = float("inf")

    def g(u_):
        return u_ - center + lam * grad(u_)

    for it in range(max_newton):
        gu = g(u)
        res_norm = _norm(gu)
        if res_norm <= tol:
            success = True
            return u, IronfiMFInnerStats(
                n_newton=it,
                n_cg_total=n_cg_total,
                final_residual_norm=res_norm,
                success=True,
                elapsed_s=time.time() - t0,
            )

        def Ax(v):
            return v + lam * hvp(u, v)

        s, cg_stats = cg_solve(Ax, -gu, tol=cg_tol, max_it=cg_max_it)
        n_cg_total += cg_stats.n_cg

        if not backtracking:
            u = u + s
            continue

        base = res_norm if np.isfinite(res_norm) else np.inf
        step = 1.0
        accepted = False
        for _ in range(max_ls):
            u_try = u + step * s
            r_try = _norm(g(u_try))
            if np.isfinite(r_try) and r_try <= base:
                u = u_try
                accepted = True
                break
            step *= beta
        if not accepted:
            u = u + step * s

    return u, IronfiMFInnerStats(
        n_newton=max_newton,
        n_cg_total=n_cg_total,
        final_residual_norm=res_norm,
        success=success,
        elapsed_s=time.time() - t0,
    )


def ironfi_step_matrix_free(
    *,
    x: np.ndarray,
    v: np.ndarray,
    alpha: float,
    mu: float,
    gamma: float,
    rng: np.random.Generator,
    grad: Callable[[np.ndarray], np.ndarray],
    hvp: Callable[[np.ndarray, np.ndarray], np.ndarray],
    inner_tol: float,
    inner_max_newton: int,
    cg_tol: float,
    cg_max_it: int,
    sigma_center: float = 0.0,
    update_gamma_flag: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
    """
    Matrix-free IRON-FI step: inner solve via Newton-CG.

    By default sigma_center=0 and noise comes from minibatches (stochastic gradients).

    Parameters
    ----------
    x, v : np.ndarray
        Outer states.
    alpha : float
        Outer stepsize parameter.
    mu, gamma : float
        Damping parameters (algorithmic; used to form tau and lambda).
    rng : np.random.Generator
        RNG (only used if sigma_center != 0).
    grad : callable
        Gradient oracle (typically minibatch gradient).
    hvp : callable
        Hessian-vector product oracle (typically minibatch HVP).
    inner_tol : float
        Inner residual tolerance ||g(u)||.
    inner_max_newton : int
        Max Newton iterations.
    cg_tol, cg_max_it : float, int
        CG solve settings per Newton step.
    sigma_center : float, default=0.0
        Optional center noise scale (set to 0 when using minibatch noise).
    update_gamma_flag : bool, default=True
        Whether to update gamma by implicit Euler.

    Returns
    -------
    x_next, v_next : np.ndarray
        Updated states.
    gamma_next : float
        Updated gamma.
    info : dict
        Contains tau, lambda, and inner solver stats.
    """
    tau = (1.0 / alpha) + (mu / gamma)
    lam = alpha / (gamma * (1.0 + tau))
    c = (v + tau * x) / (1.0 + tau)
    if sigma_center != 0.0:
        xi = (np.sqrt(alpha) / (1.0 + tau)) * sigma_center * rng.normal(size=x.shape)
    else:
        xi = 0.0
    center = c + xi

    x_next, inner = solve_resolvent_newton_cg(
        u0=x,
        center=center,
        lam=lam,
        grad=grad,
        hvp=hvp,
        tol=inner_tol,
        max_newton=inner_max_newton,
        cg_tol=cg_tol,
        cg_max_it=cg_max_it,
        backtracking=True,
    )

    v_next = x_next + (x_next - x) / alpha
    gamma_next = (gamma + alpha * mu) / (1.0 + alpha) if update_gamma_flag else gamma
    info = {"tau": tau, "lam": lam, "inner": inner}
    return x_next, v_next, gamma_next, info

