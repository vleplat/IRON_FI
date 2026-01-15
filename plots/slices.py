import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

# ---------- Stable primitives for nonconvex model ----------

def logcosh_stable(x):
    ax = np.abs(x)
    # log cosh x = |x| + log(1 + exp(-2|x|)) - log 2   (no overflow)
    return ax + np.log1p(np.exp(-2.0 * ax)) - np.log(2.0)


def tanh_stable(x):
    return np.tanh(x)


def sech2_stable(x):
    t = tanh_stable(x)
    return 1.0 - t * t

# ---------- Objectives ----------

def f_quadratic(x, A, b):
    # x shape (..., 3); A (3x3), b (3,)
    r = A @ x.T - b[:, None]  # (3, npts)
    return 0.5 * np.sum(r * r, axis=0)  # (npts,)


def f_logcosh(x, A, b):
    # u = logcosh(x) elementwise
    u = logcosh_stable(x)
    r = A @ u.T - b[:, None]
    return 0.5 * np.sum(r * r, axis=0)

# ---------- Grid & plotting ----------

def _make_grid(xmin, xmax, ymin, ymax, n=150, pad=0.05):
    # Expand a bit for nicer margins
    dx, dy = xmax - xmin, ymax - ymin
    xmin -= pad * dx; xmax += pad * dx
    ymin -= pad * dy; ymax += pad * dy
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    return XX, YY


def _auto_limits_from_clouds(X_init, X_final, dims):
    # dims: tuple of two indices for axes; take robust limits (1%..99%)
    pts = np.concatenate([X_init[dims, :], X_final[dims, :]], axis=1)
    lo = np.percentile(pts, 1.0, axis=1)
    hi = np.percentile(pts, 99.0, axis=1)
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def _contour_panel(ax, A, b, objective, XX, YY, fixed_coord, fixed_val,
                   X_init, X_final, dims, x_star=None, title=None, cmap='jet', levels=15,
                   norm=None, vmin=None, vmax=None, draw_lines=False):
    # Build grid points in R^3 with one coordinate fixed
    # dims = (i, j) → vary x_i, x_j; fix k ≠ i,j at fixed_val
    n = XX.size
    grid = np.zeros((3, n))
    grid[dims[0], :] = XX.ravel()
    grid[dims[1], :] = YY.ravel()
    grid[fixed_coord, :] = fixed_val
    vals = objective(grid.T, A, b).reshape(XX.shape)

    # Contours
    cs = ax.contourf(XX, YY, vals, levels=levels, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=0.85)
    if draw_lines:
        ax.contour(XX, YY, vals, levels=levels, colors='gray', linewidths=0.2, alpha=0.25)
    # Projected clouds
    # Use marker/color not present in jet colormap (black/white), distinct markers
    ax.scatter(X_init[dims[0], :], X_init[dims[1], :], s=6, c='white', edgecolors='black', linewidths=0.3,
               alpha=0.9, marker='^', label='init', zorder=3)
    # Final: flashy magenta with white edge to pop over jet
    ax.scatter(X_final[dims[0], :], X_final[dims[1], :], s=10, c='#ff00ff', edgecolors='white', linewidths=0.4,
               alpha=0.95, marker='o', label='final', zorder=3)

    # Mark x* if provided (quadratic)
    if x_star is not None:
        ax.plot([x_star[dims[0]]], [x_star[dims[1]]], marker='*', markersize=10, color='red', label='$x^*$')

    ax.set_xlabel(f'$x_{dims[0]+1}$')
    ax.set_ylabel(f'$x_{dims[1]+1}$')
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    if title:
        ax.set_title(title)
    return cs, vals.min(), vals.max()


def plot_objective_slices(
    A, b, X_init, X_final,
    model='quadratic',            # 'quadratic' or 'logcosh'
    x_star=None,                  # provide for quadratic; None for nonconvex
    fixed_strategy='median_final',# or 'xstar' (quadratic)
    ngrid=150,
    savepath=None,
    suptitle=None,
    fixed_lim=None,               # if set (float or (xmin,xmax,ymin,ymax)), use fixed [-L,L] per axis
    cmap='jet',                   # requested blue→red scale
    levels=15,                    # limit the number of contour levels
    scale='linear',               # 'linear' or 'log'
    vclip=(1, 99)                 # percentile clip for vmin/vmax (ignored if None)
):
    """
    Produces a 3-panel figure:
      Panel 1: vary (x1,x2), fix x3
      Panel 2: vary (x1,x3), fix x2
      Panel 3: vary (x2,x3), fix x1
    and overlays initial/final point clouds (projected).
    """
    assert X_init.shape[0] == 3 and X_final.shape[0] == 3, "Expect 3D points."

    # Choose objective
    if model == 'quadratic':
        objective = f_quadratic
    elif model == 'logcosh':
        objective = f_logcosh
    else:
        raise ValueError("model must be 'quadratic' or 'logcosh'")

    # Choose fixed coordinate values
    if fixed_strategy == 'xstar':
        if x_star is None:
            raise ValueError("x_star required when fixed_strategy='xstar'")
        fixed_vals = [float(x_star[2]), float(x_star[1]), float(x_star[0])]
    else:
        # median of final along each coord
        fixed_vals = [
            float(np.median(X_final[2, :])),
            float(np.median(X_final[1, :])),
            float(np.median(X_final[0, :]))
        ]

    # Prepare panels
    pairs = [(0,1), (0,2), (1,2)]
    fixed_coords = [2, 1, 0]
    titles = [r'$(x_1,x_2)$ (fix $x_3$)', r'$(x_1,x_3)$ (fix $x_2$)', r'$(x_2,x_3)$ (fix $x_1$)']

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), dpi=150, constrained_layout=True)

    # Build contours with consistent color scale (use combined vmin/vmax)
    XXs, YYs = [], []
    limits = []
    for (i, j), k in zip(pairs, fixed_coords):
        if fixed_lim is not None:
            if isinstance(fixed_lim, (int, float)):
                xmin = ymin = -float(fixed_lim)
                xmax = ymax = float(fixed_lim)
            else:
                xmin, xmax, ymin, ymax = fixed_lim
        else:
            xmin, xmax, ymin, ymax = _auto_limits_from_clouds(X_init, X_final, (i, j))
        limits.append((xmin, xmax, ymin, ymax))
        XX, YY = _make_grid(xmin, xmax, ymin, ymax, n=ngrid)
        XXs.append(XX); YYs.append(YY)

    # First pass: compute all vals for global vmin/vmax (for consistent colormap)
    all_vals_samples = []
    for (i, j), k, XX, YY, fval in zip(pairs, fixed_coords, XXs, YYs, fixed_vals):
        grid = np.zeros((3, XX.size))
        grid[i, :] = XX.ravel()
        grid[j, :] = YY.ravel()
        grid[k, :] = fval
        vals = objective(grid.T, A, b)
        # downsample to reduce memory if huge
        if vals.size > 50000:
            vals = vals[:: max(1, vals.size // 50000)]
        all_vals_samples.append(vals)
    all_vals = np.concatenate(all_vals_samples) if all_vals_samples else np.array([0.0])
    if vclip is not None:
        vmin = np.percentile(all_vals, vclip[0])
        vmax = np.percentile(all_vals, vclip[1])
    else:
        vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    # Cap the maximum objective value to improve contrast
    vmax = min(vmax, 20.0)
    # For log scaling ensure positive bounds
    eps = 1e-9
    if scale == 'log':
        vmin = max(vmin, eps)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Second pass: draw panels
    for ax, (i, j), k, XX, YY, lims, title, fval in zip(
        axs, pairs, fixed_coords, XXs, YYs, limits, titles, fixed_vals
    ):
        cs, vmn, vmx = _contour_panel(
            ax, A, b, objective, XX, YY, k, fval,
            X_init, X_final, (i, j), x_star=x_star if model=='quadratic' else None,
            title=title, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, levels=levels
        )
        ax.set_xlim(lims[0], lims[1]); ax.set_ylim(lims[2], lims[3])

    # One legend for all
    handles, labels = axs[0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.02))

    # Single colorbar
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, fraction=0.025, pad=0.04)
    cbar.ax.set_ylabel('objective value', rotation=90)

    if suptitle:
        fig.suptitle(suptitle, y=1.04)

    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
    return fig, axs
