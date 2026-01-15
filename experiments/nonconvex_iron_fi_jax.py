# Nonconvex (log-cosh) IRON-FI prototype (JAX)
import argparse, sys, os
try:
    import jax
    import jax.numpy as jnp
    from jax import vmap
    from jax import lax
except Exception as e:
    print("JAX is required for this experiment. Install jax/jaxlib.\n", e)
    sys.exit(1)

import numpy as np
import matplotlib.pyplot as plt
from plots.slices import plot_objective_slices

# Stable primitives

def logcosh_stable(x):
    ax = jnp.abs(x)
    return ax + jnp.log1p(jnp.exp(-2.0 * ax)) - jnp.log(2.0)


def tanh_stable(x):
    return jnp.tanh(x)


def sech2_stable(x):
    t = tanh_stable(x)
    return 1.0 - t * t


def grad_f(Q, c, x):
    u = logcosh_stable(x)
    g = (Q @ u) - c[:, None]
    return g * tanh_stable(x)


def hess_f_single(Q, c, x_single, use_gauss_newton: bool):
    u  = logcosh_stable(x_single)
    t  = tanh_stable(x_single)
    s2 = sech2_stable(x_single)
    g  = (Q @ u) - c
    diag_term  = jnp.diag(g * s2)
    if use_gauss_newton:
        return diag_term
    outer_term = Q * jnp.outer(t, t)
    return diag_term + outer_term


def newton_solve(Q, c, cxi, lam, x0, max_it=10, tol=1e-10, step_cap=1.0, use_gauss_newton=False):
    n, m = x0.shape
    vhess = vmap(lambda u: hess_f_single(Q, c, u, use_gauss_newton), in_axes=1, out_axes=2)

    def step(u):
        gval = u - cxi + lam * grad_f(Q, c, u)  # g(u)=0
        H = jnp.eye(n)[:, :, None] + lam * vhess(u)
        delta = jax.vmap(lambda Hm, gm: jnp.linalg.solve(Hm, gm))(jnp.moveaxis(H, 2, 0),
                                                                   jnp.moveaxis(gval, 1, 0))
        delta = jnp.moveaxis(delta, 0, 1)
        # step cap per sample
        norms = jnp.linalg.norm(delta, axis=0)
        scale = jnp.minimum(1.0, step_cap / (norms + 1e-12))
        delta = delta * scale
        return u - delta

    def loop_body(carry, _):
        u = carry
        u_next = step(u)
        return u_next, None

    u = x0
    u_prev = u
    u, _ = lax.scan(loop_body, u, None, length=max_it)
    # simple convergence check after fixed iters (keeps JAX-friendly control flow)
    return u


def run_once(args, alpha_scale):
    key = jax.random.PRNGKey(args.seed)
    n = 3
    A = jnp.diag(jnp.array([1.0, 1.0, 5.0]))
    Q = A.T @ A
    b = jnp.ones((n,)) * 3.0
    c = A.T @ b

    mu = float(jnp.min(jnp.linalg.eigvalsh(Q)))
    L  = float(jnp.max(jnp.linalg.eigvalsh(Q)))
    gamma = jnp.sqrt(mu)
    alpha = 0.6 * (2.0 / np.sqrt(np.linalg.cond(np.array(Q))))
    alpha = float(alpha * alpha_scale)

    ns = args.nsamples
    key, kx = jax.random.split(key)
    x = jax.random.normal(kx, (n, ns))
    v = jnp.zeros((n, ns))

    res = []
    keep = min(5000, ns)
    X_init = np.array(x[:, :keep])

    for _ in range(args.iters):
        tau = (1.0/alpha) + (mu / float(gamma))
        lam = alpha / (float(gamma) * (1.0 + tau))
        c_k = (v + tau * x) / (1.0 + tau)

        key, kn = jax.random.split(key)
        eta = jax.random.normal(kn, (n, ns))
        xi  = (jnp.sqrt(alpha) / (1.0 + tau)) * args.sigma * eta
        cxi = c_k + xi

        x_next = newton_solve(Q, c, cxi, lam, x, max_it=args.newton_it, tol=args.tol,
                              step_cap=args.step_cap, use_gauss_newton=args.gauss_newton)
        v = (x_next - x) / alpha + x_next
        x = x_next
        gamma = (gamma + alpha * mu) / (1.0 + alpha)

        res.append(float(jnp.linalg.norm(jnp.mean(x, axis=1))))

    X_final = np.array(x[:, -keep:])

    # plots
    fig1 = plt.figure(figsize=(5.5, 3), dpi=150)
    plt.semilogy(res)
    plt.grid(True)
    plt.xlabel('Iteration k')
    plt.ylabel(r'$\\|\\bar x^{(k)}\\|_2$')
    if args.save_figs:
        os.makedirs('figs', exist_ok=True)
        fig1.savefig(f"figs/ncx_alpha{int(alpha_scale)}_mean_norm.pdf", bbox_inches="tight")
    if not args.no_show:
        plt.show()
    plt.close(fig1)

    fig2, axs = plt.subplots(1, 3, figsize=(7.5, 2.8), dpi=150)
    pairs = [(0,1), (0,2), (1,2)]
    for ax, (i,j) in zip(axs, pairs):
        ax.scatter(X_init[i],  X_init[j],  s=6, c='white', edgecolors='black', linewidths=0.3, alpha=0.9, marker='^', label='Initial')
        ax.scatter(X_final[i], X_final[j], s=10, c='#ff00ff', edgecolors='white', linewidths=0.4, alpha=0.95, marker='o', label='Final')
        ax.set_aspect('equal', 'box'); ax.grid(True); ax.legend(fontsize=7)
        ax.set_xlim(-args.plot_lim, args.plot_lim); ax.set_ylim(-args.plot_lim, args.plot_lim)
        ax.set_title(f'Coords ({i+1},{j+1})')
    plt.tight_layout()
    if args.save_figs:
        fig2.savefig(f"figs/ncx_alpha{int(alpha_scale)}_cloud.pdf", bbox_inches="tight")
    if not args.no_show:
        plt.show()
    plt.close(fig2)

    # slices (use numpy-based utility)
    if args.save_figs:
        plot_objective_slices(
            np.array(A), np.array(b), np.array(X_init), np.array(X_final),
            model='logcosh', x_star=None, fixed_strategy='median_final', ngrid=150,
            savepath=f"figs/logcosh_slices_alpha{int(alpha_scale)}.pdf",
            suptitle=None, fixed_lim=args.plot_lim, cmap='jet', levels=15
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsamples', type=int, default=4096)
    parser.add_argument('--iters', type=int, default=15)
    parser.add_argument('--alpha-scale', type=float, nargs='+', default=[100.0])
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--newton-it', type=int, default=10)
    parser.add_argument('--tol', type=float, default=1e-10)
    parser.add_argument('--step-cap', type=float, default=1.0)
    parser.add_argument('--gauss-newton', action='store_true')
    parser.add_argument('--plot-lim', type=float, default=5.0)
    parser.add_argument('--save-figs', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    for s in args.alpha_scale:
        run_once(args, s)

    print("\n=== Summary (nonconvex prototype, JAX) ===")
    print(f"nsamples={args.nsamples}, iters={args.iters}, sigma={args.sigma}")
    print(f"alpha_scales={args.alpha_scale}, newton_it={args.newton_it}, tol={args.tol}, step_cap={args.step_cap}, gauss_newton={args.gauss_newton}")


if __name__ == '__main__':
    main()
