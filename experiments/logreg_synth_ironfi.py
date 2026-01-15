import argparse
import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from ironfi.ironfi import ironfi_step
from utils.expio import append_csv_row, make_run_dir
from utils.plotting import savefig


def sigmoid(t: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    # For t >= 0: 1/(1+exp(-t)) is safe
    # For t < 0: exp(t)/(1+exp(t)) avoids exp(-t) overflow
    out = np.empty_like(t, dtype=float)
    pos = t >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
    et = np.exp(t[~pos])
    out[~pos] = et / (1.0 + et)
    return out


def _matmul_checked(A: np.ndarray, B: np.ndarray, name: str) -> np.ndarray:
    """
    Matmul with warnings suppressed (some BLAS backends can raise spurious FP warnings)
    and explicit finite checks to still catch true numerical blow-ups.
    """
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        out = A @ B
    if not np.all(np.isfinite(out)):
        raise FloatingPointError(f"Non-finite values encountered in {name}.")
    return out


def make_synth_logreg(rng: np.random.Generator, n: int, d: int, w_scale: float = 1.0):
    X = rng.normal(size=(n, d))
    w_star = w_scale * rng.normal(size=(d,))
    p = sigmoid(_matmul_checked(X, w_star, "X @ w_star"))
    y01 = (rng.random(size=(n,)) < p).astype(np.int64)
    y = 2 * y01 - 1  # {-1,+1}
    return X, y, w_star


def loss(w: np.ndarray, X: np.ndarray, y: np.ndarray, reg: float) -> float:
    z = y * _matmul_checked(X, w, "X @ w (loss)")
    # stable: log(1+exp(-z))
    return float(np.mean(np.logaddexp(0.0, -z)) + 0.5 * reg * np.dot(w, w))


def grad(w: np.ndarray, X: np.ndarray, y: np.ndarray, reg: float) -> np.ndarray:
    z = y * _matmul_checked(X, w, "X @ w (grad)")
    s = sigmoid(-z)
    g = -_matmul_checked(X.T, (y * s), "X.T @ (y*s) (grad)") / X.shape[0]
    return g + reg * w


def hess(w: np.ndarray, X: np.ndarray, y: np.ndarray, reg: float) -> np.ndarray:
    z = y * _matmul_checked(X, w, "X @ w (hess)")
    s = sigmoid(-z)
    d = s * (1.0 - s)  # in (0, 1/4)
    H = _matmul_checked(X.T, (d[:, None] * X), "X.T @ (D*X) (hess)") / X.shape[0]
    H.flat[:: H.shape[0] + 1] += reg
    return H


def solve_w_hat_star(X: np.ndarray, y: np.ndarray, reg: float, max_it: int = 50, tol: float = 1e-10) -> np.ndarray:
    """Deterministic minimizer via Newton on full data (SPD Hessian due to reg)."""
    d = X.shape[1]
    w = np.zeros(d)
    for _ in range(max_it):
        g = grad(w, X, y, reg)
        if float(np.linalg.norm(g)) < tol:
            break
        H = hess(w, X, y, reg)
        try:
            from scipy.linalg import cho_factor, cho_solve
            cf = cho_factor(H, overwrite_a=False, check_finite=False)
            step = cho_solve(cf, -g, check_finite=False)
        except Exception:
            step = np.linalg.solve(H, -g)
        w = w + step
    return w


def run_single(
    *,
    alpha: float,
    inner_tol: float,
    args,
    X: np.ndarray,
    y: np.ndarray,
    w_hat_star: np.ndarray,
    rng: np.random.Generator,
    run_dir: str,
) -> tuple[float, float]:
    d = X.shape[1]
    mu = args.reg
    gamma = float(np.sqrt(mu))
    x = rng.normal(size=(d,))
    v = np.zeros((d,))

    burn = int(args.burn_frac * args.iters)
    mse_sum = 0.0
    mse_count = 0
    inner_it_sum = 0.0
    inner_it_count = 0
    t_start = time.time()

    metrics_path = os.path.join(run_dir, f"metrics_alpha{alpha:g}_tol{inner_tol:g}.csv")
    header = [
        "k", "alpha", "tol", "loss", "mse",
        "inner_newton", "inner_res", "inner_success",
        "elapsed_s",
    ]

    for k in range(args.iters):
        x, v, gamma, info = ironfi_step(
            x=x,
            v=v,
            alpha=alpha,
            mu=mu,
            gamma=gamma,
            sigma=args.sigma,
            rng=rng,
            grad=lambda w: grad(w, X, y, args.reg),
            hess=lambda w: hess(w, X, y, args.reg),
            inner_tol=inner_tol,
            inner_max_it=args.inner_max_it,
            update_gamma_flag=True,
            inner_use_cholesky=True,
        )

        inner = info["inner"]
        mse = float(np.dot(x - w_hat_star, x - w_hat_star))
        fval = loss(x, X, y, args.reg)

        append_csv_row(
            metrics_path,
            {
                "k": k,
                "alpha": alpha,
                "tol": inner_tol,
                "loss": fval,
                "mse": mse,
                "inner_newton": inner.n_newton,
                "inner_res": inner.final_residual_norm,
                "inner_success": int(inner.success),
                "elapsed_s": time.time() - t_start,
            },
            header=header,
        )

        if k >= burn:
            mse_sum += mse
            mse_count += 1
            inner_it_sum += inner.n_newton
            inner_it_count += 1

    mse_hat = mse_sum / max(1, mse_count)
    inner_it_hat = inner_it_sum / max(1, inner_it_count)
    return mse_hat, inner_it_hat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--d", type=int, default=50)
    parser.add_argument("--reg", type=float, default=1e-2)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--burn-frac", type=float, default=0.3)
    parser.add_argument("--inner-max-it", type=int, default=20)
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=[1, 2, 5, 10, 20, 50, 100, 200])
    parser.add_argument("--tol-grid", type=float, nargs="+", default=[1e-2, 1e-4, 1e-6])
    parser.add_argument("--slope-fit-min-alpha", type=float, default=5.0)
    parser.add_argument("--run-prefix", type=str, default="logreg_synth_ironfi")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X, y, w_true = make_synth_logreg(rng, args.n, args.d, w_scale=1.0)
    w_hat_star = solve_w_hat_star(X, y, args.reg)

    config = {
        "n": args.n,
        "d": args.d,
        "reg": args.reg,
        "sigma": args.sigma,
        "seed": args.seed,
        "iters": args.iters,
        "burn_frac": args.burn_frac,
        "inner_max_it": args.inner_max_it,
        "alpha_grid": args.alpha_grid,
        "tol_grid": args.tol_grid,
    }
    run_dir = make_run_dir("logs", args.run_prefix, config)

    results: dict[str, list[tuple[float, float, float]]] = {}
    for tol in args.tol_grid:
        results[str(tol)] = []
        for alpha in args.alpha_grid:
            cond_rng = np.random.default_rng(args.seed + 1000 + int(10_000 * alpha) + int(1e6 * tol))
            mse_hat, inner_it_hat = run_single(
                alpha=alpha,
                inner_tol=tol,
                args=args,
                X=X,
                y=y,
                w_hat_star=w_hat_star,
                rng=cond_rng,
                run_dir=run_dir,
            )
            results[str(tol)].append((alpha, mse_hat, inner_it_hat))

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    os.makedirs("figs", exist_ok=True)

    # Plot 1: stationary MSE vs alpha (best tol)
    best_tol = min(args.tol_grid)
    arr = results[str(best_tol)]
    alpha_vals = np.array([t[0] for t in arr], dtype=float)
    mse_vals = np.array([t[1] for t in arr], dtype=float)

    # Log-log slope fit: log(MSE) ≈ a + b log(alpha)  => b ~ -1
    mask = (alpha_vals >= float(args.slope_fit_min_alpha)) & (mse_vals > 0) & np.isfinite(mse_vals)
    slope = float("nan")
    if int(np.sum(mask)) >= 2:
        slope = float(np.polyfit(np.log(alpha_vals[mask]), np.log(mse_vals[mask]), deg=1)[0])
    print(f"[slope] best_tol={best_tol:g}, fit alpha>= {args.slope_fit_min_alpha:g}: slope={slope:.3f}")

    with open(os.path.join(run_dir, "slopes.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_tol": float(best_tol),
                "slope_fit_min_alpha": float(args.slope_fit_min_alpha),
                "slope": slope,
                "alpha_used": alpha_vals[mask].tolist(),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.2), dpi=150)
    ax.loglog(alpha_vals, mse_vals, "o-", label=f"tol={best_tol:g}")
    if np.isfinite(slope):
        ax.text(
            0.05,
            0.05,
            f"slope≈{slope:.2f} (alpha≥{args.slope_fit_min_alpha:g})",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.8),
        )
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("stationary MSE")
    ax.set_title("Synthetic logreg: stationary MSE vs alpha (IRON-FI)")
    ax.legend()
    savefig(f"figs/synth_logreg_mse_vs_alpha_tol{best_tol:g}.pdf")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    # Plot 2: tolerance effect
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.2), dpi=150)
    for tol in args.tol_grid:
        arr = results[str(tol)]
        a = np.array([t[0] for t in arr], dtype=float)
        mse = np.array([t[1] for t in arr], dtype=float)
        ax.loglog(a, mse, "o-", label=f"tol={tol:g}")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("stationary MSE")
    ax.set_title("Synthetic logreg: tolerance effect (IRON-FI)")
    ax.legend()
    savefig("figs/synth_logreg_tol_effect.pdf")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    # Plot 3: mean inner iterations vs alpha
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.2), dpi=150)
    for tol in args.tol_grid:
        arr = results[str(tol)]
        a = np.array([t[0] for t in arr], dtype=float)
        itmean = np.array([t[2] for t in arr], dtype=float)
        ax.semilogx(a, itmean, "o-", label=f"tol={tol:g}")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("mean inner Newton iters (stationary)")
    ax.set_title("Synthetic logreg: inner cost vs alpha (IRON-FI)")
    ax.legend()
    savefig("figs/synth_logreg_inner_iters_vs_alpha.pdf")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

