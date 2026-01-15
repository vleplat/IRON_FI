# Quadratic IRON-FI experiment (NumPy/SciPy)
import argparse
import numpy as np
from numpy.linalg import cond
import matplotlib.pyplot as plt
from scipy.linalg import qr
from ironfi.resolvent import ironfi_params, ironfi_quadratic_step
from ironfi.noise import sample_gaussian_xi
from ironfi.gamma import update_gamma
from plots.utils import plot_mean_error, scatter_projections


def run_once(eigs, bval, nsamples, iters, alpha_scale, sigma, seed, save_prefix=None, show=True):
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
    gamma = np.sqrt(mu)
    alpha1 = 2.0 / np.sqrt(cond(A))
    alpha2 = (mu + gamma + 2*np.sqrt(gamma*L)) / (L - mu) if L > mu else alpha1
    alpha  = max(alpha1, alpha2) * float(alpha_scale)

    # Init
    x = rng.normal(size=(n, nsamples))
    v = np.zeros((n, nsamples))

    # logging
    keep = min(10000, nsamples)
    X_init = x[:, :keep].copy()
    res = []

    for _ in range(iters):
        tau, lam = ironfi_params(alpha, mu, gamma)
        xi = sample_gaussian_xi(rng, sigma, alpha, tau, x.shape)
        x, v, gamma, info = ironfi_quadratic_step(A, b, x, v, alpha, mu, gamma, xi)
        res.append(np.linalg.norm(np.mean(x, axis=1) - x_star))

    X_final = x[:, -keep:].copy()

    # plots (display the requested alpha-scale in the legend for clarity)
    fig1, ax1 = plot_mean_error(res, label=f'alpha={alpha_scale}')
    if save_prefix:
        fig1.savefig(f"figs/quad_mean_alpha{int(alpha_scale)}.pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig1)

    fig2, axs2 = scatter_projections(X_init, X_final, x_star=x_star, lim=8)
    if save_prefix:
        fig2.savefig(f"figs/quad_clouds_alpha{int(alpha_scale)}.pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha-scale', type=float, nargs='+', default=[1000.0])
    parser.add_argument('--nsamples', type=int, default=200000)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eigs', type=float, nargs='+', default=[1.0, 1.0, 3.0])
    parser.add_argument('--bval', type=float, default=5.0)
    parser.add_argument('--save-figs', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    if args.save_figs:
        import os
        os.makedirs('figs', exist_ok=True)

    show = not args.no_show

    for s in args.alpha_scale:
        prefix = None
        if args.save_figs:
            prefix = f"quad_alpha{int(s)}"
        run_once(args.eigs, args.bval, args.nsamples, args.iters, s, args.sigma, args.seed, save_prefix=prefix, show=show)


if __name__ == '__main__':
    main()
