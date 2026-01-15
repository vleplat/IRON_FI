import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from datasets.mnist import load_mnist
from experiments.mnist_softmax_core import run_adamw, run_ironfi, run_naggs
from utils.expio import make_run_dir


def mean_std(x: np.ndarray) -> Tuple[float, float]:
    """
    Compute mean and sample standard deviation.

    Parameters
    ----------
    x : np.ndarray
        1D array of values.

    Returns
    -------
    mean : float
        Mean of x.
    std : float
        Sample standard deviation (ddof=1). Returns 0.0 if x has length 1.
    """
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    return mean, std


def plot_mean_std(ax, x, y_mean, y_std, label):
    ax.plot(x, y_mean, label=label)
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/mnist")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[128, 256, 384])
    p.add_argument("--ironfi-alpha-by-batch", type=float, nargs="+", default=[1.0, 1.5, 2.5])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--reg", type=float, default=1e-4)
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--run-prefix", type=str, default="mnist_softmax_journal")

    # Baselines
    p.add_argument("--adamw-lr", type=float, default=1e-3)
    p.add_argument("--adamw-weight-decay", type=float, default=0.0)
    p.add_argument("--naggs-alpha", type=float, default=0.5)
    p.add_argument("--naggs-mu", type=float, default=1.0)
    p.add_argument("--naggs-gamma0", type=float, default=1.0)

    # IRON-FI
    p.add_argument("--ironfi-mu", type=float, default=1.0)
    p.add_argument("--ironfi-gamma0", type=float, default=1.0)
    p.add_argument("--ironfi-inner-tol", type=float, default=1e-3)
    p.add_argument("--ironfi-inner-newton", type=int, default=8)
    p.add_argument("--ironfi-cg-tol", type=float, default=1e-3)
    p.add_argument("--ironfi-cg-max-it", type=int, default=200)

    args = p.parse_args()
    if len(args.ironfi_alpha_by_batch) != len(args.batch_sizes):
        raise ValueError("--ironfi-alpha-by-batch must have same length as --batch-sizes")

    Xtr, ytr, Xte, yte = load_mnist(args.data_dir, download=False, flatten=True, normalize=True)
    os.makedirs("figs", exist_ok=True)

    config = vars(args)
    run_dir = make_run_dir("logs", args.run_prefix, config)

    summary: Dict[str, Dict] = {}

    for bs, alpha_iron in zip(args.batch_sizes, args.ironfi_alpha_by_batch):
        key = f"batch{bs}"
        summary[key] = {"batch_size": bs, "ironfi_alpha": alpha_iron, "seeds": args.seeds}

        # collect per-seed curves
        adam_tr, adam_te, adam_time = [], [], []
        nag_tr, nag_te, nag_time = [], [], []
        iron_tr, iron_te, iron_time = [], [], []
        iron_in, iron_icg, iron_ir = [], [], []

        for seed in args.seeds:
            tag = f"b{bs}_s{seed}"

            t0 = time.time()
            c_adam = run_adamw(
                Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                reg=args.reg, epochs=args.epochs, batch_size=bs, seed=seed,
                lr=args.adamw_lr, weight_decay=args.adamw_weight_decay,
                run_dir=run_dir, tag=tag,
            )
            adam_time.append(time.time() - t0)
            adam_tr.append(c_adam.train_loss)
            adam_te.append(c_adam.test_acc)

            t0 = time.time()
            c_nag = run_naggs(
                Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                reg=args.reg, epochs=args.epochs, batch_size=bs, seed=seed,
                alpha=args.naggs_alpha, mu=args.naggs_mu, gamma0=args.naggs_gamma0,
                run_dir=run_dir, tag=tag,
            )
            nag_time.append(time.time() - t0)
            nag_tr.append(c_nag.train_loss)
            nag_te.append(c_nag.test_acc)

            t0 = time.time()
            c_iron = run_ironfi(
                Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                reg=args.reg, epochs=args.epochs, batch_size=bs, seed=seed,
                alpha=alpha_iron, alpha2=None, alpha2_start_epoch=None,
                mu=args.ironfi_mu, gamma0=args.ironfi_gamma0,
                inner_tol=args.ironfi_inner_tol, inner_newton=args.ironfi_inner_newton,
                cg_tol=args.ironfi_cg_tol, cg_max_it=args.ironfi_cg_max_it,
                run_dir=run_dir, tag=tag,
            )
            iron_time.append(time.time() - t0)
            iron_tr.append(c_iron.train_loss)
            iron_te.append(c_iron.test_acc)
            iron_in.append(c_iron.inner_newton)
            iron_icg.append(c_iron.inner_cg)
            iron_ir.append(c_iron.inner_res)

        # stack
        adam_te = np.stack(adam_te, axis=0)
        nag_te = np.stack(nag_te, axis=0)
        iron_te = np.stack(iron_te, axis=0)

        # final epoch acc stats
        adam_final = adam_te[:, -1]
        nag_final = nag_te[:, -1]
        iron_final = iron_te[:, -1]

        summary[key]["final_acc"] = {
            "adamw": {"mean": mean_std(adam_final)[0], "std": mean_std(adam_final)[1]},
            "nag_gs": {"mean": mean_std(nag_final)[0], "std": mean_std(nag_final)[1]},
            "ironfi": {"mean": mean_std(iron_final)[0], "std": mean_std(iron_final)[1]},
        }
        summary[key]["time_s"] = {
            "adamw": {"mean": mean_std(np.array(adam_time))[0], "std": mean_std(np.array(adam_time))[1]},
            "nag_gs": {"mean": mean_std(np.array(nag_time))[0], "std": mean_std(np.array(nag_time))[1]},
            "ironfi": {"mean": mean_std(np.array(iron_time))[0], "std": mean_std(np.array(iron_time))[1]},
        }

        # averaged curves plots
        e = np.arange(args.epochs)

        def mstd(curves_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
            A = np.stack(curves_list, axis=0)
            return np.mean(A, axis=0), np.std(A, axis=0, ddof=1) if A.shape[0] > 1 else (np.mean(A, axis=0), np.zeros_like(np.mean(A, axis=0)))

        adam_tr_m, adam_tr_s = mstd(adam_tr)
        nag_tr_m, nag_tr_s = mstd(nag_tr)
        iron_tr_m, iron_tr_s = mstd(iron_tr)

        adam_te_m, adam_te_s = mstd([c for c in adam_te])
        nag_te_m, nag_te_s = mstd([c for c in nag_te])
        iron_te_m, iron_te_s = mstd([c for c in iron_te])

        fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.6), dpi=150)
        plot_mean_std(ax, e, adam_tr_m, adam_tr_s, f"AdamW lr={args.adamw_lr:g}")
        plot_mean_std(ax, e, nag_tr_m, nag_tr_s, f"NAG-GS alpha={args.naggs_alpha:g}")
        plot_mean_std(ax, e, iron_tr_m, iron_tr_s, f"IRON-FI alpha={alpha_iron:g}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("train loss")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"figs/mnist_journal_train_loss_batch{bs}.pdf")
        if not args.no_show:
            plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.6), dpi=150)
        plot_mean_std(ax, e, adam_te_m, adam_te_s, f"AdamW lr={args.adamw_lr:g}")
        plot_mean_std(ax, e, nag_te_m, nag_te_s, f"NAG-GS alpha={args.naggs_alpha:g}")
        plot_mean_std(ax, e, iron_te_m, iron_te_s, f"IRON-FI alpha={alpha_iron:g}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("test accuracy")
        ax.set_ylim(0.87, 0.94)
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"figs/mnist_journal_test_acc_batch{bs}.pdf")
        if not args.no_show:
            plt.show()
        plt.close(fig)

        iron_in_m, iron_in_s = mstd(iron_in)
        iron_icg_m, iron_icg_s = mstd(iron_icg)
        iron_ir_m, iron_ir_s = mstd(iron_ir)

        fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.6), dpi=150)
        plot_mean_std(ax, e, iron_in_m, iron_in_s, "IRON: mean Newton/epoch")
        plot_mean_std(ax, e, iron_icg_m, iron_icg_s, "IRON: mean CG iters/epoch")
        plot_mean_std(ax, e, iron_ir_m, iron_ir_s, "IRON: mean residual/epoch")
        ax.set_xlabel("epoch")
        ax.set_ylabel("inner stats (IRON-FI)")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"figs/mnist_journal_ironfi_inner_batch{bs}.pdf")
        if not args.no_show:
            plt.show()
        plt.close(fig)

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("wrote summary:", os.path.join(run_dir, "summary.json"))


if __name__ == "__main__":
    main()

