import argparse
import json
import os
import numpy as np
from numpy.linalg import cond
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov
from ironfi.resolvent import ironfi_params, ironfi_quadratic_step
from ironfi.noise import sample_gaussian_xi
from plots.utils import plot_quadratic_mse_history, plot_stationary_scaled_mse, scatter_projections


def _quadratic_metrics(x: np.ndarray, x_star: np.ndarray) -> dict[str, float]:
    diff = x - x_star[:, None]
    mean = np.mean(x, axis=1)
    centered = x - mean[:, None]
    mean_error = float(np.linalg.norm(mean - x_star))
    bias_sq = float(np.dot(mean - x_star, mean - x_star))
    mse = float(np.mean(np.sum(diff * diff, axis=0)))
    cov_trace = max(mse - bias_sq, 0.0)
    empirical_cov = centered @ centered.T / x.shape[1]
    return {
        "mean_error": mean_error,
        "bias_sq": bias_sq,
        "mse": mse,
        "cov_trace": cov_trace,
        "mean": mean,
        "cov": empirical_cov,
    }


def _stationary_quadratic_prediction(A: np.ndarray, alpha: float, mu: float, gamma: float, sigma: float) -> dict[str, np.ndarray | float]:
    n = A.shape[0]
    tau, lam = ironfi_params(alpha, mu, gamma)
    s = 1.0 + tau
    R = np.linalg.solve(np.eye(n) + lam * A, np.eye(n))
    top_left = (tau / s) * R
    top_right = (1.0 / s) * R
    bot_left = (1.0 + 1.0 / alpha) * top_left - (1.0 / alpha) * np.eye(n)
    bot_right = (1.0 + 1.0 / alpha) * top_right
    M = np.block([[top_left, top_right], [bot_left, bot_right]])
    G = np.vstack([R, (1.0 + 1.0 / alpha) * R])
    xi_var = alpha * sigma * sigma / (s * s)
    Q = xi_var * (G @ G.T)
    P = solve_discrete_lyapunov(M, Q)
    cov_x = P[:n, :n]
    asymptotic_constant = float((gamma * gamma) * (sigma * sigma) * np.trace(np.linalg.inv(A @ A)))
    return {
        "cov_x": cov_x,
        "mse": float(np.trace(cov_x)),
        "scaled_mse": float(alpha * np.trace(cov_x)),
        "asymptotic_constant": asymptotic_constant,
    }


def run_once(
    eigs,
    bval,
    nsamples,
    iters,
    alpha_scale,
    sigma,
    seed,
    burn_frac=0.3,
    gamma_mode="fixed",
    save_prefix=None,
    show=True,
):
    rng = np.random.default_rng(seed)

    # Problem
    eigs = np.array(eigs, dtype=float)
    n = len(eigs)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    A = Q @ np.diag(eigs) @ Q.T
    mu = np.min(eigs)
    L  = np.max(eigs)
    b = np.ones(n) * bval
    x_star = np.linalg.solve(A, b)

    # Params
    gamma_init = float(np.sqrt(mu))
    gamma = gamma_init
    alpha1 = 2.0 / np.sqrt(cond(A))
    alpha2 = (mu + gamma + 2*np.sqrt(gamma*L)) / (L - mu) if L > mu else alpha1
    alpha  = max(alpha1, alpha2) * float(alpha_scale)

    # Init
    x = rng.normal(size=(n, nsamples))
    v = np.zeros((n, nsamples))

    # logging
    keep = min(10000, nsamples)
    X_init = x[:, :keep].copy()
    mean_error_hist = []
    mse_hist = []
    bias_sq_hist = []
    cov_trace_hist = []

    for _ in range(iters):
        tau, lam = ironfi_params(alpha, mu, gamma)
        xi = sample_gaussian_xi(rng, sigma, alpha, tau, x.shape)
        x, v, gamma_next, info = ironfi_quadratic_step(A, b, x, v, alpha, mu, gamma, xi)
        gamma = gamma_next if gamma_mode == "updated" else gamma_init
        metrics = _quadratic_metrics(x, x_star)
        mean_error_hist.append(metrics["mean_error"])
        mse_hist.append(metrics["mse"])
        bias_sq_hist.append(metrics["bias_sq"])
        cov_trace_hist.append(metrics["cov_trace"])

    X_final = x[:, -keep:].copy()
    final_metrics = _quadratic_metrics(X_final, x_star)
    burn_start = min(iters - 1, max(0, int(burn_frac * iters)))
    stationary_summary = {
        "alpha": float(alpha),
        "alpha_scale": float(alpha_scale),
        "gamma_mode": gamma_mode,
        "burn_start": int(burn_start),
        "stationary_mean_error": float(np.mean(mean_error_hist[burn_start:])),
        "stationary_mse": float(np.mean(mse_hist[burn_start:])),
        "stationary_bias_sq": float(np.mean(bias_sq_hist[burn_start:])),
        "stationary_cov_trace": float(np.mean(cov_trace_hist[burn_start:])),
    }

    exact_prediction = None
    if gamma_mode == "fixed":
        exact_prediction = _stationary_quadratic_prediction(A, alpha, mu, gamma_init, sigma)
        stationary_summary["exact_stationary_mse"] = exact_prediction["mse"]
        stationary_summary["exact_scaled_mse"] = exact_prediction["scaled_mse"]
        stationary_summary["asymptotic_constant"] = exact_prediction["asymptotic_constant"]

    # plots
    fig1, ax1 = plot_quadratic_mse_history(
        mean_error_hist,
        mse_hist,
        bias_sq_hist,
        cov_trace_hist,
        label=f"alpha={alpha_scale:g}",
    )
    if save_prefix:
        fig1.savefig(f"figs/quad_mean_alpha{int(alpha_scale)}.pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig1)

    fig2, axs2 = scatter_projections(
        X_init,
        X_final,
        x_star=x_star,
        lim=8,
    )
    if save_prefix:
        fig2.savefig(f"figs/quad_clouds_alpha{int(alpha_scale)}.pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig2)

    return {
        "alpha": float(alpha),
        "alpha_scale": float(alpha_scale),
        "mu": float(mu),
        "L": float(L),
        "gamma_init": gamma_init,
        "gamma_mode": gamma_mode,
        "sigma": float(sigma),
        "stationary_summary": stationary_summary,
        "history": {
            "mean_error": [float(v) for v in mean_error_hist],
            "mse": [float(v) for v in mse_hist],
            "bias_sq": [float(v) for v in bias_sq_hist],
            "cov_trace": [float(v) for v in cov_trace_hist],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha-scale', type=float, nargs='+', default=[1000.0])
    parser.add_argument('--nsamples', type=int, default=200000)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eigs', type=float, nargs='+', default=[1.0, 1.0, 3.0])
    parser.add_argument('--bval', type=float, default=5.0)
    parser.add_argument('--burn-frac', type=float, default=0.3)
    parser.add_argument('--gamma-mode', choices=['fixed', 'updated'], default='fixed')
    parser.add_argument('--save-figs', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    if args.save_figs:
        os.makedirs('figs', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

    show = not args.no_show
    summaries = []

    for s in args.alpha_scale:
        prefix = None
        if args.save_figs:
            prefix = f"quad_alpha{int(s)}"
        result = run_once(
            args.eigs,
            args.bval,
            args.nsamples,
            args.iters,
            s,
            args.sigma,
            args.seed,
            burn_frac=args.burn_frac,
            gamma_mode=args.gamma_mode,
            save_prefix=prefix,
            show=show,
        )
        summaries.append(result)

    if len(summaries) >= 2:
        alpha_vals = np.array([item["alpha"] for item in summaries], dtype=float)
        scaled_empirical = np.array(
            [item["stationary_summary"]["alpha"] * item["stationary_summary"]["stationary_mse"] for item in summaries],
            dtype=float,
        )
        scaled_exact = None
        asymptotic_constant = None
        if args.gamma_mode == 'fixed':
            scaled_exact = np.array(
                [item["stationary_summary"]["exact_scaled_mse"] for item in summaries],
                dtype=float,
            )
            asymptotic_constant = float(summaries[0]["stationary_summary"]["asymptotic_constant"])
        else:
            asymptotic_constant = float('nan')

        fig, ax = plot_stationary_scaled_mse(
            alpha_vals,
            scaled_empirical,
            asymptotic_constant,
            scaled_exact=scaled_exact,
        )
        if args.save_figs:
            fig.savefig("figs/quad_stationary_scaled_mse.pdf", bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    if args.save_figs:
        summary_path = os.path.join("logs", f"quad_iron_fi_summary_seed{args.seed}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "alpha_scale": args.alpha_scale,
                    "nsamples": args.nsamples,
                    "iters": args.iters,
                    "sigma": args.sigma,
                    "seed": args.seed,
                    "eigs": args.eigs,
                    "bval": args.bval,
                    "burn_frac": args.burn_frac,
                    "gamma_mode": args.gamma_mode,
                    "results": summaries,
                },
                f,
                indent=2,
                sort_keys=True,
            )


if __name__ == '__main__':
    main()
