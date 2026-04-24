import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from plots.slices import plot_objective_slices

# Problem: f(x) uses u_i = log cosh(x_i)
# ∇f(x) = [Q u - c] ⊙ tanh(x),   Q = A^T A, c = A^T b
# ∇^2 f(x) = diag( (Q u - c) ⊙ sech^2(x) ) + Q ⊙ (tanh(x) tanh(x)^T)

def stable_log_cosh(x: np.ndarray) -> np.ndarray:
    # log cosh(x) = logaddexp(x, -x) - log(2)
    return np.logaddexp(x, -x) - np.log(2.0)


def grad_f(Q: np.ndarray, c: np.ndarray, x: np.ndarray) -> np.ndarray:
    # x: (n, m)
    u = stable_log_cosh(x)
    g = (Q @ u) - c[:, None]
    return g * np.tanh(x)

def grad_f_single(Q: np.ndarray, c: np.ndarray, x_single: np.ndarray) -> np.ndarray:
    # x_single: (n,)
    u = stable_log_cosh(x_single)
    g = (Q @ u) - c
    return g * np.tanh(x_single)


def hess_f_batch(Q: np.ndarray, c: np.ndarray, x: np.ndarray, use_gauss_newton: bool) -> np.ndarray:
    # x: (n, m) -> H: (m, n, n)
    n, m = x.shape
    u = stable_log_cosh(x)              # (n, m)
    t = np.tanh(x)                      # (n, m)
    s2 = 1.0 - t * t                    # sech^2 via identity to avoid overflow
    g = (Q @ u) - c[:, None]            # (n, m)

    # diag term per sample
    dvec = (g * s2).T                   # (m, n)
    diag_term = np.zeros((m, n, n), dtype=x.dtype)
    idx = np.arange(n)
    diag_term[:, idx, idx] = dvec

    if use_gauss_newton:
        return diag_term
    # hadamard term Q ⊙ (t t^T) per sample
    outer = np.einsum('im,jm->mij', t, t)  # (m, n, n)
    hadamard = outer * Q[None, :, :]
    return diag_term + hadamard


def newton_solve(Q: np.ndarray, c: np.ndarray, cxi: np.ndarray, lam: float, x0: np.ndarray,
                 max_it: int = 10, tol: float = 1e-10, chunk_size: int = 512,
                 step_cap: float = 1.0, max_ls: int = 8, clip_x: float = None,
                 use_gauss_newton: bool = False) -> np.ndarray:
    # Solve g(u) = u - cxi + lam ∇f(u) = 0 with damping/backtracking
    u = x0.copy()
    n, m = u.shape
    for _ in range(max_it):
        u_prev = u
        gval = u - cxi + lam * grad_f(Q, c, u)            # (n, m)
        Hs = hess_f_batch(Q, c, u, use_gauss_newton)      # (m, n, n)
        # Solve per chunk to avoid batched-solve quirks
        for start in range(0, m, chunk_size):
            end = min(start + chunk_size, m)
            M = np.eye(n)[None, :, :] + lam * Hs[start:end, :, :]   # (k, n, n)
            rhs = gval[:, start:end].T                               # (k, n)
            # Solve each sample
            for k in range(end - start):
                idx = start + k
                delta_k = np.linalg.solve(M[k], rhs[k])              # (n,)
                # trust-region cap
                dn = np.linalg.norm(delta_k)
                if dn > step_cap:
                    delta_k = (step_cap / dn) * delta_k
                # backtracking on residual norm
                res0 = np.linalg.norm(gval[:, idx])
                s = 1.0
                accepted = False
                for _ls in range(max_ls):
                    u_try = u[:, idx] - s * delta_k
                    # optional clipping
                    if clip_x is not None:
                        u_try = np.clip(u_try, -clip_x, clip_x)
                    # compute residual norm for this sample only
                    res_try = np.linalg.norm(u_try - cxi[:, idx] + lam * grad_f_single(Q, c, u_try))
                    if res_try <= (1 - 1e-4 * s) * res0:
                        u[:, idx] = u_try
                        accepted = True
                        break
                    s *= 0.5
                if not accepted:
                    u[:, idx] = u[:, idx] - s * delta_k
                    if clip_x is not None:
                        u[:, idx] = np.clip(u[:, idx], -clip_x, clip_x)
        if np.max(np.linalg.norm(u - u_prev, axis=0)) < tol:
            break
    return u


def _plot_pairwise_clouds(X_ref: np.ndarray, X_last: np.ndarray, X_pool: np.ndarray, plot_lim: float):
    fig, axs = plt.subplots(2, 3, figsize=(8.2, 5.4), dpi=150)
    pairs = [(0, 1), (0, 2), (1, 2)]
    row_titles = ["Last iterate", "Late-time pooled"]
    targets = [X_last, X_pool]
    for row, (row_title, target) in enumerate(zip(row_titles, targets)):
        for ax, (i, j) in zip(axs[row], pairs):
            ax.plot(X_ref[i], X_ref[j], "x", ms=1, alpha=0.35, label="Initial")
            ax.plot(target[i], target[j], "o", ms=1, alpha=0.35, label=row_title)
            ax.set_aspect("equal", "box")
            ax.grid(True)
            ax.set_xlim(-plot_lim, plot_lim)
            ax.set_ylim(-plot_lim, plot_lim)
            ax.set_title(f"{row_title}: coords ({i+1},{j+1})", fontsize=8)
            if row == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def _finite_columns(X: np.ndarray) -> np.ndarray:
    mask = np.all(np.isfinite(X), axis=0)
    return X[:, mask] if np.any(mask) else X[:, :1].copy()


def run_once(nsamples: int, iters: int, alpha_scale: float, sigma: float, seed: int,
             newton_it: int, tol: float, save_figs: bool, no_show: bool,
             step_cap: float, max_ls: int, clip_x: float, gauss_newton: bool,
             plot_lim: float, burn_frac: float, gamma_mode: str):
    rng = np.random.default_rng(seed)

    # Problem setup (n=3)
    n = 3
    A = np.diag([1.0, 1.0, 5.0])
    Q = A.T @ A
    b = np.ones(n) * 3.0
    c = A.T @ b

    mu = float(np.min(np.linalg.eigvalsh(Q)))
    L  = float(np.max(np.linalg.eigvalsh(Q)))
    gamma = np.sqrt(mu)
    alpha = 0.6 * (2.0 / np.sqrt(np.linalg.cond(Q)))
    alpha = float(alpha * alpha_scale)

    ns = nsamples
    x = rng.normal(size=(n, ns))
    v = np.zeros((n, ns))
    gamma_init = gamma

    res = []
    keep = min(5000, ns)
    X_init = x[:, :keep].copy()
    burn_start = min(iters - 1, max(0, int(burn_frac * iters)))
    pooled_samples = []

    for k in range(iters):
        tau = (1.0 / alpha) + (mu / float(gamma))
        lam = alpha / (float(gamma) * (1.0 + tau))
        c_k = (v + tau * x) / (1.0 + tau)

        eta = rng.normal(size=(n, ns))
        xi  = (np.sqrt(alpha) / (1.0 + tau)) * sigma * eta
        cxi = c_k + xi

        x_next = newton_solve(
            Q, c, cxi, lam, x,
            max_it=newton_it, tol=tol,
            step_cap=step_cap, max_ls=max_ls, clip_x=clip_x,
            use_gauss_newton=gauss_newton,
        )
        v = (x_next - x) / alpha + x_next
        x = x_next
        gamma_next = (gamma + alpha * mu) / (1.0 + alpha)
        gamma = gamma_next if gamma_mode == "updated" else gamma_init

        res.append(float(np.linalg.norm(np.mean(x, axis=1))))
        if k >= burn_start:
            pooled_samples.append(x[:, :keep].copy())

    X_last = _finite_columns(x[:, :keep].copy())
    X_pool = _finite_columns(np.concatenate(pooled_samples, axis=1) if pooled_samples else X_last.copy())

    # plots
    fig1 = plt.figure(figsize=(5.5, 3), dpi=150)
    plt.semilogy(res)
    plt.grid(True)
    plt.xlabel('Iteration k')
    plt.ylabel(r'$\|\bar x^{(k)}\|_2$')
    plt.title('Nonconvex IRON-FI (illustrative mean-iterate proxy)')
    if save_figs:
        os.makedirs('figs', exist_ok=True)
        fig1.savefig(f"figs/ncx_numpy_alpha{int(alpha_scale)}_mean_norm.pdf", bbox_inches="tight")
    if not no_show:
        plt.show()
    plt.close(fig1)

    fig2 = _plot_pairwise_clouds(X_init, X_last, X_pool, plot_lim)
    if save_figs:
        fig2.savefig(f"figs/ncx_numpy_alpha{int(alpha_scale)}_cloud.pdf", bbox_inches="tight")
    if not no_show:
        plt.show()
    plt.close(fig2)

    # Objective slices with projected clouds
    if save_figs:
        plot_objective_slices(
            A, b, X_init, X_pool,
            model='logcosh', x_star=None,
            fixed_strategy='median_final', ngrid=150,
            savepath=f"figs/logcosh_slices_alpha{int(alpha_scale)}.pdf",
            suptitle=None,
            fixed_lim=plot_lim
        )

    summary = {
        "alpha": float(alpha),
        "alpha_scale": float(alpha_scale),
        "sigma": float(sigma),
        "seed": int(seed),
        "iters": int(iters),
        "burn_start": int(burn_start),
        "gamma_mode": gamma_mode,
        "pooled_sample_count": int(X_pool.shape[1]),
        "late_mean_norm_proxy": float(np.mean(res[burn_start:])),
    }
    if save_figs:
        os.makedirs("logs", exist_ok=True)
        with open(os.path.join("logs", f"nonconvex_alpha{int(alpha_scale)}_seed{seed}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsamples', type=int, default=2048)
    parser.add_argument('--iters', type=int, default=15)
    parser.add_argument('--alpha-scale', type=float, nargs='+', default=[100.0])
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--newton-it', type=int, default=10)
    parser.add_argument('--tol', type=float, default=1e-10)
    parser.add_argument('--step-cap', type=float, default=1.0)
    parser.add_argument('--max-ls', type=int, default=8)
    parser.add_argument('--clip-x', type=float, default=30.0)
    parser.add_argument('--plot-lim', type=float, default=15.0)
    parser.add_argument('--burn-frac', type=float, default=0.5)
    parser.add_argument('--gamma-mode', choices=['fixed', 'updated'], default='updated')
    parser.add_argument('--gauss-newton', action='store_true')
    parser.add_argument('--save-figs', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    for s in args.alpha_scale:
        run_once(
            args.nsamples, args.iters, s, args.sigma, args.seed,
            args.newton_it, args.tol, args.save_figs, args.no_show,
            args.step_cap, args.max_ls, args.clip_x, args.gauss_newton,
            float(args.plot_lim), args.burn_frac, args.gamma_mode,
        )


if __name__ == '__main__':
    main()
