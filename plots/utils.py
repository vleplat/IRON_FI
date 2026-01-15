import matplotlib.pyplot as plt
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

def scatter_projections(X_init, X_final, x_star=None, lim=8):
    n = X_init.shape[0]
    pairs = [(0,1), (0,2), (1,2)] if n >= 3 else [(0,1)]
    fig, axs = plt.subplots(1, len(pairs), figsize=(7.5, 2.8), dpi=150)
    if len(pairs) == 1: axs = [axs]
    for ax, (i, j) in zip(axs, pairs):
        ax.plot(X_init[i], X_init[j], 'x', ms=1, alpha=0.5, label='Initial')
        ax.plot(X_final[i], X_final[j], 'o', ms=1, alpha=0.5, label='Final')
        if x_star is not None and len(x_star) > max(i,j):
            ax.plot([x_star[i]], [x_star[j]], '*', ms=8)
        ax.set_aspect('equal', 'box'); ax.grid(True); ax.legend(fontsize=7)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(f'Coords ({i+1},{j+1})')
    fig.tight_layout()
    return fig, axs
