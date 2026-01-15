import matplotlib.pyplot as plt
import numpy as np


def savefig(path: str, dpi: int = 150) -> None:
    """
    Save the current Matplotlib figure to disk.

    Parameters
    ----------
    path : str
        Output path (directories must exist).
    dpi : int, default=150
        DPI for raster elements.
    """
    plt.savefig(path, bbox_inches="tight", dpi=dpi)


def plot_loglog_mse_vs_alpha(alpha_vals, mse_vals, label=None):
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.2), dpi=150)
    ax.loglog(alpha_vals, mse_vals, "o-", label=label)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"stationary MSE")
    if label:
        ax.legend()
    return fig, ax


def plot_inner_iters_vs_alpha(alpha_vals, iters_mean, label=None):
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.2), dpi=150)
    ax.semilogx(alpha_vals, iters_mean, "o-", label=label)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("mean inner Newton iterations (stationary)")
    if label:
        ax.legend()
    return fig, ax

