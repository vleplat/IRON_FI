import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

def plot_mean_error(res, alpha=None, label=None):
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3), dpi=150)
    label_str = label if label is not None else (fr'$\alpha={alpha:.2e}$' if alpha is not None else None)
    if label_str is not None:
        ax.semilogy(res, label=label_str)
    else:
        ax.semilogy(res)
    ax.set_xlabel('Iteration $k$'); ax.set_ylabel(r'$\|\bar x^{(k)}-x^*\|_2$')
    ax.grid(True); ax.legend(); ax.set_title('IRON-FI: mean error decay')
    return fig, ax

def _add_covariance_ellipse(ax, mean, cov, i, j, color, label, n_std=2.0, lw=1.5, ls='-'):
    subcov = np.asarray(cov)[np.ix_([i, j], [i, j])]
    if not np.all(np.isfinite(subcov)):
        return
    evals, evecs = np.linalg.eigh(subcov)
    evals = np.maximum(evals, 0.0)
    if np.allclose(evals, 0.0):
        return
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    angle = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])))
    width = 2.0 * n_std * np.sqrt(evals[0])
    height = 2.0 * n_std * np.sqrt(evals[1])
    ellipse = Ellipse(
        xy=(mean[i], mean[j]),
        width=width,
        height=height,
        angle=angle,
        facecolor='none',
        edgecolor=color,
        linewidth=lw,
        linestyle=ls,
        label=label,
    )
    ax.add_patch(ellipse)

def scatter_projections(X_init, X_final, x_star=None, lim=8, empirical_cov=None, reference_cov=None, ellipse_n_std=2.0):
    n = X_init.shape[0]
    pairs = [(0,1), (0,2), (1,2)] if n >= 3 else [(0,1)]
    fig, axs = plt.subplots(1, len(pairs), figsize=(7.5, 2.8), dpi=150)
    if len(pairs) == 1: axs = [axs]
    final_mean = np.mean(X_final, axis=1)
    for ax, (i, j) in zip(axs, pairs):
        ax.plot(X_init[i], X_init[j], 'x', ms=1, alpha=0.5, label='Initial')
        ax.plot(X_final[i], X_final[j], 'o', ms=1, alpha=0.5, label='Final')
        if x_star is not None and len(x_star) > max(i,j):
            ax.plot([x_star[i]], [x_star[j]], '*', ms=8, label='$x^*$')
        if empirical_cov is not None:
            _add_covariance_ellipse(
                ax,
                final_mean,
                empirical_cov,
                i,
                j,
                color='tab:green',
                label='Empirical cov.',
                n_std=ellipse_n_std,
                ls='-',
            )
        if reference_cov is not None:
            center = x_star if x_star is not None else final_mean
            _add_covariance_ellipse(
                ax,
                center,
                reference_cov,
                i,
                j,
                color='tab:red',
                label='Lyapunov cov.',
                n_std=ellipse_n_std,
                ls='--',
            )
        ax.set_aspect('equal', 'box'); ax.grid(True); ax.legend(fontsize=7)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(f'Coords ({i+1},{j+1})')
    fig.tight_layout()
    return fig, axs

def plot_quadratic_mse_history(mean_error, mse, bias_sq, cov_trace, label=None):
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.4), dpi=150)
    label_suffix = f" ({label})" if label else ""
    ax.semilogy(mean_error, label=r'$\|\bar x_k-x^\star\|_2$' + label_suffix)
    ax.semilogy(mse, label=r'$\widehat{\mathrm{MSE}}_k$' + label_suffix)
    ax.semilogy(bias_sq, label=r'$\|\bar x_k-x^\star\|_2^2$' + label_suffix)
    ax.semilogy(cov_trace, label=r'$\mathrm{tr}(\widehat{\mathrm{Cov}}(x_k))$' + label_suffix)
    ax.set_xlabel('Iteration $k$')
    ax.set_ylabel('quadratic error metric')
    ax.grid(True)
    ax.legend(fontsize=8)
    ax.set_title('IRON-FI quadratic: ensemble MSE decomposition')
    fig.tight_layout()
    return fig, ax

def plot_stationary_scaled_mse(alpha_vals, scaled_empirical, asymptotic_constant, scaled_exact=None):
    fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.4), dpi=150)
    ax.semilogx(alpha_vals, scaled_empirical, 'o-', label=r'empirical $\alpha\,\widehat{\mathrm{MSE}}_\infty$')
    if scaled_exact is not None:
        ax.semilogx(alpha_vals, scaled_exact, 's--', label=r'exact Lyapunov $\alpha\,\mathrm{MSE}_\infty$')
    ax.axhline(asymptotic_constant, color='tab:red', linestyle=':', label=r'$C_{\mathrm{quad}}$')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\alpha \cdot \widehat{\mathrm{MSE}}_\infty$')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=8)
    ax.set_title('Quadratic stationary constant check')
    fig.tight_layout()
    return fig, ax
