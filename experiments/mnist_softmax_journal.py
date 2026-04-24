import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from datasets.mnist import load_mnist
from experiments.mnist_softmax_core import RunCurves, run_adamw, run_ironfi, run_naggs
from utils.expio import make_run_dir


def mean_std(x: np.ndarray) -> Tuple[float, float]:
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    return mean, std


def plot_mean_std(ax, x, y_mean, y_std, label):
    ax.plot(x, y_mean, label=label)
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)


def mstd(curves_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    A = np.stack(curves_list, axis=0)
    mean = np.mean(A, axis=0)
    std = np.std(A, axis=0, ddof=1) if A.shape[0] > 1 else np.zeros_like(mean)
    return mean, std


def stratified_train_val_split(X: np.ndarray, y: np.ndarray, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        cls_idx = rng.permutation(cls_idx)
        n_val = max(1, int(round(val_frac * cls_idx.size)))
        val_idx.append(cls_idx[:n_val])
        train_idx.append(cls_idx[n_val:])
    train_idx = rng.permutation(np.concatenate(train_idx))
    val_idx = rng.permutation(np.concatenate(val_idx))
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def evaluate_candidates(candidates: List[float], runner, tune_seeds: List[int], metric_name: str) -> tuple[float, list[dict[str, float]]]:
    records = []
    best_value = None
    best_score = -np.inf
    for value in candidates:
        scores = []
        for seed in tune_seeds:
            curves = runner(value, seed)
            scores.append(float(getattr(curves, metric_name)[-1]))
        score_mean = float(np.mean(scores))
        records.append({"value": float(value), "mean_final_val_acc": score_mean, "scores": [float(s) for s in scores]})
        if score_mean > best_score:
            best_score = score_mean
            best_value = float(value)
    return float(best_value), records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/mnist")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--tune-epochs", type=int, default=10)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[128, 256, 384])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--tune-seeds", type=int, nargs="+", default=None)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=123)
    p.add_argument("--reg", type=float, default=1e-4)
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--run-prefix", type=str, default="mnist_softmax_journal")

    # Symmetric tuning grids
    p.add_argument("--adamw-lr-grid", type=float, nargs="+", default=[3e-4, 5e-4, 7e-4, 1e-3, 1.5e-3, 2e-3, 3e-3])
    p.add_argument("--adamw-weight-decay", type=float, default=0.0)
    p.add_argument("--naggs-alpha-grid", type=float, nargs="+", default=[0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0])
    p.add_argument("--naggs-mu", type=float, default=1.0)
    p.add_argument("--naggs-gamma0", type=float, default=1.0)
    p.add_argument("--ironfi-alpha-grid", type=float, nargs="+", default=[0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])

    # IRON-FI
    p.add_argument("--ironfi-mu", type=float, default=1.0)
    p.add_argument("--ironfi-gamma0", type=float, default=1.0)
    p.add_argument("--ironfi-inner-tol", type=float, default=1e-3)
    p.add_argument("--ironfi-inner-newton", type=int, default=8)
    p.add_argument("--ironfi-cg-tol", type=float, default=1e-3)
    p.add_argument("--ironfi-cg-max-it", type=int, default=200)

    args = p.parse_args()
    tune_seeds = args.tune_seeds if args.tune_seeds is not None else args.seeds

    Xtrain_full, ytrain_full, Xtest, ytest = load_mnist(args.data_dir, download=False, flatten=True, normalize=True)
    Xtrain, ytrain, Xval, yval = stratified_train_val_split(Xtrain_full, ytrain_full, args.val_frac, args.split_seed)
    os.makedirs("figs", exist_ok=True)

    config = vars(args).copy()
    config["tune_seeds"] = tune_seeds
    config["train_size"] = int(Xtrain.shape[0])
    config["val_size"] = int(Xval.shape[0])
    config["test_size"] = int(Xtest.shape[0])
    run_dir = make_run_dir("logs", args.run_prefix, config)

    summary: Dict[str, Dict] = {
        "split": {
            "train_size": int(Xtrain.shape[0]),
            "val_size": int(Xval.shape[0]),
            "test_size": int(Xtest.shape[0]),
            "val_frac": float(args.val_frac),
            "split_seed": int(args.split_seed),
        }
    }

    for bs in args.batch_sizes:
        key = f"batch{bs}"
        summary[key] = {"batch_size": bs, "seeds": args.seeds, "tune_seeds": tune_seeds}

        def adam_tune_runner(lr: float, seed: int) -> RunCurves:
            return run_adamw(
                Xtr=Xtrain,
                ytr=ytrain,
                Xval=Xval,
                yval=yval,
                Xtest=None,
                ytest=None,
                reg=args.reg,
                epochs=args.tune_epochs,
                batch_size=bs,
                seed=seed,
                lr=lr,
                weight_decay=args.adamw_weight_decay,
                run_dir=run_dir,
                tag=f"tune_b{bs}_adam_s{seed}",
            )

        def nag_tune_runner(alpha: float, seed: int) -> RunCurves:
            return run_naggs(
                Xtr=Xtrain,
                ytr=ytrain,
                Xval=Xval,
                yval=yval,
                Xtest=None,
                ytest=None,
                reg=args.reg,
                epochs=args.tune_epochs,
                batch_size=bs,
                seed=seed,
                alpha=alpha,
                mu=args.naggs_mu,
                gamma0=args.naggs_gamma0,
                run_dir=run_dir,
                tag=f"tune_b{bs}_nag_s{seed}",
            )

        def iron_tune_runner(alpha: float, seed: int) -> RunCurves:
            return run_ironfi(
                Xtr=Xtrain,
                ytr=ytrain,
                Xval=Xval,
                yval=yval,
                Xtest=None,
                ytest=None,
                reg=args.reg,
                epochs=args.tune_epochs,
                batch_size=bs,
                seed=seed,
                alpha=alpha,
                alpha2=None,
                alpha2_start_epoch=None,
                mu=args.ironfi_mu,
                gamma0=args.ironfi_gamma0,
                inner_tol=args.ironfi_inner_tol,
                inner_newton=args.ironfi_inner_newton,
                cg_tol=args.ironfi_cg_tol,
                cg_max_it=args.ironfi_cg_max_it,
                run_dir=run_dir,
                tag=f"tune_b{bs}_iron_s{seed}",
            )

        best_adamw_lr, adam_tuning = evaluate_candidates(args.adamw_lr_grid, adam_tune_runner, tune_seeds, "val_acc")
        best_naggs_alpha, nag_tuning = evaluate_candidates(args.naggs_alpha_grid, nag_tune_runner, tune_seeds, "val_acc")
        best_ironfi_alpha, iron_tuning = evaluate_candidates(args.ironfi_alpha_grid, iron_tune_runner, tune_seeds, "val_acc")

        summary[key]["selected_hparams"] = {
            "adamw_lr": best_adamw_lr,
            "naggs_alpha": best_naggs_alpha,
            "ironfi_alpha": best_ironfi_alpha,
        }
        summary[key]["tuning"] = {
            "adamw": adam_tuning,
            "nag_gs": nag_tuning,
            "ironfi": iron_tuning,
            "tune_epochs": int(args.tune_epochs),
            "selection_metric": "final validation accuracy",
        }

        adam_tr, adam_test, adam_time = [], [], []
        nag_tr, nag_test, nag_time = [], [], []
        iron_tr, iron_test, iron_time = [], [], []
        iron_in, iron_icg, iron_ir = [], [], []

        for seed in args.seeds:
            tag = f"b{bs}_s{seed}"

            c_adam = run_adamw(
                Xtr=Xtrain_full,
                ytr=ytrain_full,
                Xval=Xtest,
                yval=ytest,
                Xtest=Xtest,
                ytest=ytest,
                reg=args.reg,
                epochs=args.epochs,
                batch_size=bs,
                seed=seed,
                lr=best_adamw_lr,
                weight_decay=args.adamw_weight_decay,
                run_dir=run_dir,
                tag=tag,
            )
            adam_tr.append(c_adam.train_loss)
            adam_test.append(c_adam.test_acc)
            adam_time.append(c_adam.elapsed_s)

            c_nag = run_naggs(
                Xtr=Xtrain_full,
                ytr=ytrain_full,
                Xval=Xtest,
                yval=ytest,
                Xtest=Xtest,
                ytest=ytest,
                reg=args.reg,
                epochs=args.epochs,
                batch_size=bs,
                seed=seed,
                alpha=best_naggs_alpha,
                mu=args.naggs_mu,
                gamma0=args.naggs_gamma0,
                run_dir=run_dir,
                tag=tag,
            )
            nag_tr.append(c_nag.train_loss)
            nag_test.append(c_nag.test_acc)
            nag_time.append(c_nag.elapsed_s)

            c_iron = run_ironfi(
                Xtr=Xtrain_full,
                ytr=ytrain_full,
                Xval=Xtest,
                yval=ytest,
                Xtest=Xtest,
                ytest=ytest,
                reg=args.reg,
                epochs=args.epochs,
                batch_size=bs,
                seed=seed,
                alpha=best_ironfi_alpha,
                alpha2=None,
                alpha2_start_epoch=None,
                mu=args.ironfi_mu,
                gamma0=args.ironfi_gamma0,
                inner_tol=args.ironfi_inner_tol,
                inner_newton=args.ironfi_inner_newton,
                cg_tol=args.ironfi_cg_tol,
                cg_max_it=args.ironfi_cg_max_it,
                run_dir=run_dir,
                tag=tag,
            )
            iron_tr.append(c_iron.train_loss)
            iron_test.append(c_iron.test_acc)
            iron_time.append(c_iron.elapsed_s)
            iron_in.append(c_iron.inner_newton)
            iron_icg.append(c_iron.inner_cg)
            iron_ir.append(c_iron.inner_res)

        adam_test_stack = np.stack(adam_test, axis=0)
        nag_test_stack = np.stack(nag_test, axis=0)
        iron_test_stack = np.stack(iron_test, axis=0)

        summary[key]["final_acc"] = {
            "adamw": {"mean": mean_std(adam_test_stack[:, -1])[0], "std": mean_std(adam_test_stack[:, -1])[1]},
            "nag_gs": {"mean": mean_std(nag_test_stack[:, -1])[0], "std": mean_std(nag_test_stack[:, -1])[1]},
            "ironfi": {"mean": mean_std(iron_test_stack[:, -1])[0], "std": mean_std(iron_test_stack[:, -1])[1]},
        }
        summary[key]["time_s"] = {
            "adamw": {"mean": mean_std(np.array([curve[-1] for curve in adam_time]))[0], "std": mean_std(np.array([curve[-1] for curve in adam_time]))[1]},
            "nag_gs": {"mean": mean_std(np.array([curve[-1] for curve in nag_time]))[0], "std": mean_std(np.array([curve[-1] for curve in nag_time]))[1]},
            "ironfi": {"mean": mean_std(np.array([curve[-1] for curve in iron_time]))[0], "std": mean_std(np.array([curve[-1] for curve in iron_time]))[1]},
        }

        e = np.arange(args.epochs)
        adam_tr_m, adam_tr_s = mstd(adam_tr)
        nag_tr_m, nag_tr_s = mstd(nag_tr)
        iron_tr_m, iron_tr_s = mstd(iron_tr)

        adam_test_m, adam_test_s = mstd([c for c in adam_test_stack])
        nag_test_m, nag_test_s = mstd([c for c in nag_test_stack])
        iron_test_m, iron_test_s = mstd([c for c in iron_test_stack])

        adam_time_m, _ = mstd(adam_time)
        nag_time_m, _ = mstd(nag_time)
        iron_time_m, _ = mstd(iron_time)

        fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.6), dpi=150)
        plot_mean_std(ax, e, adam_tr_m, adam_tr_s, f"AdamW lr={best_adamw_lr:g}")
        plot_mean_std(ax, e, nag_tr_m, nag_tr_s, f"NAG-GS alpha={best_naggs_alpha:g}")
        plot_mean_std(ax, e, iron_tr_m, iron_tr_s, f"IRON-FI alpha={best_ironfi_alpha:g}")
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
        plot_mean_std(ax, e, adam_test_m, adam_test_s, f"AdamW lr={best_adamw_lr:g}")
        plot_mean_std(ax, e, nag_test_m, nag_test_s, f"NAG-GS alpha={best_naggs_alpha:g}")
        plot_mean_std(ax, e, iron_test_m, iron_test_s, f"IRON-FI alpha={best_ironfi_alpha:g}")
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

        fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.6), dpi=150)
        ax.plot(adam_time_m, adam_test_m, label=f"AdamW lr={best_adamw_lr:g}")
        ax.plot(nag_time_m, nag_test_m, label=f"NAG-GS alpha={best_naggs_alpha:g}")
        ax.plot(iron_time_m, iron_test_m, label=f"IRON-FI alpha={best_ironfi_alpha:g}")
        ax.set_xlabel("wall-clock time (s)")
        ax.set_ylabel("test accuracy")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"figs/mnist_journal_test_acc_vs_time_batch{bs}.pdf")
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

