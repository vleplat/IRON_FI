import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from datasets.mnist import load_mnist
from ironfi.ironfi_mf import ironfi_step_matrix_free
from ironfi.optimizers.adamw import AdamW
from ironfi.optimizers.nag_gs import nag_gs_step
from models.softmax import SoftmaxParams, accuracy, hvp as softmax_hvp, loss_and_grad
from utils.expio import append_csv_row, make_run_dir

# NOTE: For journal runs (multi-seed aggregation), use experiments/mnist_softmax_journal.py.


def iter_minibatches(indices: np.ndarray, batch_size: int):
    for i in range(0, len(indices), batch_size):
        yield indices[i : i + batch_size]


def make_fake_mnist(rng: np.random.Generator, n_train=6000, n_test=1000, d=784, C=10):
    Xtr = rng.normal(size=(n_train, d)).astype(np.float32)
    Xte = rng.normal(size=(n_test, d)).astype(np.float32)
    ytr = rng.integers(0, C, size=(n_train,), dtype=np.int64)
    yte = rng.integers(0, C, size=(n_test,), dtype=np.int64)
    return Xtr, ytr, Xte, yte


def run_adamw(
    *,
    Xtr,
    ytr,
    Xte,
    yte,
    reg,
    epochs,
    batch_size,
    seed,
    lr,
    weight_decay,
    run_dir,
):
    rng = np.random.default_rng(seed)
    d = Xtr.shape[1]
    C = int(np.max(ytr)) + 1
    params = SoftmaxParams(W=np.zeros((d, C)), b=np.zeros((C,)))
    theta = params.pack()
    opt = AdamW(lr=lr, weight_decay=weight_decay)

    rows_path = os.path.join(run_dir, f"adamw_lr{lr:g}.csv")
    header = ["epoch", "step", "train_loss", "test_acc", "elapsed_s"]
    curves = {"train_loss": [], "test_acc": []}
    t0 = time.time()
    step = 0
    for ep in range(epochs):
        perm = rng.permutation(Xtr.shape[0])
        for bidx in iter_minibatches(perm, batch_size):
            p = SoftmaxParams.unpack(theta, d, C)
            _, g = loss_and_grad(p, Xtr[bidx], ytr[bidx], reg)
            theta = opt.step(theta, g.pack())
            step += 1

        p = SoftmaxParams.unpack(theta, d, C)
        tr_loss, _ = loss_and_grad(p, Xtr[perm[: min(10000, Xtr.shape[0])]], ytr[perm[: min(10000, Xtr.shape[0])]], reg)
        te_acc = accuracy(p, Xte, yte)
        curves["train_loss"].append(tr_loss)
        curves["test_acc"].append(te_acc)
        append_csv_row(
            rows_path,
            {"epoch": ep, "step": step, "train_loss": tr_loss, "test_acc": te_acc, "elapsed_s": time.time() - t0},
            header=header,
        )
    return curves


def run_naggs(
    *,
    Xtr,
    ytr,
    Xte,
    yte,
    reg,
    epochs,
    batch_size,
    seed,
    alpha,
    mu=1.0,
    gamma0=1.0,
    run_dir=None,
):
    rng = np.random.default_rng(seed)
    d = Xtr.shape[1]
    C = int(np.max(ytr)) + 1
    # x/v are parameter vectors
    x = SoftmaxParams(W=np.zeros((d, C)), b=np.zeros((C,))).pack()
    v = x.copy()
    gamma = float(gamma0)

    rows_path = os.path.join(run_dir, f"naggs_alpha{alpha:g}.csv")
    header = ["epoch", "step", "train_loss", "test_acc", "gamma", "elapsed_s"]
    curves = {"train_loss": [], "test_acc": []}
    t0 = time.time()
    step = 0
    for ep in range(epochs):
        perm = rng.permutation(Xtr.shape[0])
        for bidx in iter_minibatches(perm, batch_size):
            def grad_at(x_next_vec):
                p = SoftmaxParams.unpack(x_next_vec, d, C)
                _, g = loss_and_grad(p, Xtr[bidx], ytr[bidx], reg)
                return g.pack()

            x, v, gamma = nag_gs_step(x=x, v=v, gamma=gamma, alpha=alpha, mu=mu, grad_at_xnext=grad_at)
            step += 1

        p = SoftmaxParams.unpack(x, d, C)
        tr_loss, _ = loss_and_grad(p, Xtr[perm[: min(10000, Xtr.shape[0])]], ytr[perm[: min(10000, Xtr.shape[0])]], reg)
        te_acc = accuracy(p, Xte, yte)
        curves["train_loss"].append(tr_loss)
        curves["test_acc"].append(te_acc)
        append_csv_row(
            rows_path,
            {"epoch": ep, "step": step, "train_loss": tr_loss, "test_acc": te_acc, "gamma": gamma, "elapsed_s": time.time() - t0},
            header=header,
        )
    return curves


def run_ironfi(
    *,
    Xtr,
    ytr,
    Xte,
    yte,
    reg,
    epochs,
    batch_size,
    seed,
    alpha,
    alpha2: float | None = None,
    alpha2_start_epoch: int | None = None,
    mu=1.0,
    gamma0=1.0,
    inner_tol=1e-3,
    inner_newton=8,
    cg_tol=1e-3,
    cg_max_it=200,
    run_dir=None,
):
    rng = np.random.default_rng(seed)
    d = Xtr.shape[1]
    C = int(np.max(ytr)) + 1
    x = SoftmaxParams(W=np.zeros((d, C)), b=np.zeros((C,))).pack()
    v = x.copy()
    gamma = float(gamma0)

    if alpha2 is not None and alpha2_start_epoch is not None:
        rows_path = os.path.join(run_dir, f"ironfi_alpha{alpha:g}_to_{alpha2:g}_at{alpha2_start_epoch}.csv")
    else:
        rows_path = os.path.join(run_dir, f"ironfi_alpha{alpha:g}.csv")
    header = ["epoch", "step", "train_loss", "test_acc", "gamma", "inner_newton", "inner_cg", "inner_res", "elapsed_s"]
    curves = {"train_loss": [], "test_acc": [], "inner_newton": [], "inner_cg": [], "inner_res": []}
    t0 = time.time()
    step = 0
    for ep in range(epochs):
        perm = rng.permutation(Xtr.shape[0])
        inner_newton_hist = []
        inner_cg_hist = []
        inner_res_hist = []
        for bidx in iter_minibatches(perm, batch_size):
            Xb = Xtr[bidx]
            yb = ytr[bidx]

            def grad_fn(theta_vec):
                p = SoftmaxParams.unpack(theta_vec, d, C)
                _, g = loss_and_grad(p, Xb, yb, reg)
                return g.pack()

            def hvp_fn(theta_vec, vec_vec):
                p = SoftmaxParams.unpack(theta_vec, d, C)
                vv = SoftmaxParams.unpack(vec_vec, d, C)
                hv = softmax_hvp(p, Xb, yb, reg, vv)
                return hv.pack()

            alpha_now = alpha
            if alpha2 is not None and alpha2_start_epoch is not None and ep >= alpha2_start_epoch:
                alpha_now = alpha2

            x, v, gamma, info = ironfi_step_matrix_free(
                x=x,
                v=v,
                alpha=alpha_now,
                mu=mu,
                gamma=gamma,
                rng=rng,
                grad=grad_fn,
                hvp=hvp_fn,
                inner_tol=inner_tol,
                inner_max_newton=inner_newton,
                cg_tol=cg_tol,
                cg_max_it=cg_max_it,
                sigma_center=0.0,  # minibatch noise only
                update_gamma_flag=True,
            )
            step += 1
            inner = info["inner"]
            inner_newton_hist.append(inner.n_newton)
            inner_cg_hist.append(inner.n_cg_total)
            inner_res_hist.append(inner.final_residual_norm)

        p = SoftmaxParams.unpack(x, d, C)
        tr_loss, _ = loss_and_grad(p, Xtr[perm[: min(10000, Xtr.shape[0])]], ytr[perm[: min(10000, Xtr.shape[0])]], reg)
        te_acc = accuracy(p, Xte, yte)
        curves["train_loss"].append(tr_loss)
        curves["test_acc"].append(te_acc)
        curves["inner_newton"].append(float(np.mean(inner_newton_hist)))
        curves["inner_cg"].append(float(np.mean(inner_cg_hist)))
        curves["inner_res"].append(float(np.mean(inner_res_hist)))
        append_csv_row(
            rows_path,
            {
                "epoch": ep,
                "step": step,
                "train_loss": tr_loss,
                "test_acc": te_acc,
                "gamma": gamma,
                "inner_newton": float(np.mean(inner_newton_hist)),
                "inner_cg": float(np.mean(inner_cg_hist)),
                "inner_res": float(np.mean(inner_res_hist)),
                "elapsed_s": time.time() - t0,
            },
            header=header,
        )
    return curves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/mnist")
    p.add_argument("--download", action="store_true")
    p.add_argument("--fake-data", action="store_true")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--reg", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-prefix", type=str, default="mnist_softmax_partB")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--fig-tag", type=str, default="")
    p.add_argument("--tune-ironfi", action="store_true")
    p.add_argument("--tune-epochs", type=int, default=5)
    p.add_argument("--ironfi-alpha-grid", type=float, nargs="+", default=None)

    # NAG-GS
    p.add_argument("--naggs-alpha", type=float, default=0.5)
    p.add_argument("--naggs-mu", type=float, default=1.0)
    p.add_argument("--naggs-gamma0", type=float, default=1.0)

    # AdamW
    p.add_argument("--adamw-lr", type=float, default=1e-3)
    p.add_argument("--adamw-weight-decay", type=float, default=0.0)

    # IRON-FI
    p.add_argument("--ironfi-alpha", type=float, default=10.0)
    p.add_argument("--ironfi-alpha2", type=float, default=None)
    p.add_argument("--ironfi-alpha2-start-epoch", type=int, default=None)
    p.add_argument("--ironfi-mu", type=float, default=1.0)
    p.add_argument("--ironfi-gamma0", type=float, default=1.0)
    p.add_argument("--ironfi-inner-tol", type=float, default=1e-3)
    p.add_argument("--ironfi-inner-newton", type=int, default=8)
    p.add_argument("--ironfi-cg-tol", type=float, default=1e-3)
    p.add_argument("--ironfi-cg-max-it", type=int, default=200)

    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.fake_data:
        Xtr, ytr, Xte, yte = make_fake_mnist(rng)
    else:
        Xtr, ytr, Xte, yte = load_mnist(args.data_dir, download=args.download, flatten=True, normalize=True)

    config = vars(args)
    run_dir = make_run_dir("logs", args.run_prefix, config)
    os.makedirs("figs", exist_ok=True)

    curves_adamw = run_adamw(
        Xtr=Xtr,
        ytr=ytr,
        Xte=Xte,
        yte=yte,
        reg=args.reg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        lr=args.adamw_lr,
        weight_decay=args.adamw_weight_decay,
        run_dir=run_dir,
    )
    curves_naggs = run_naggs(
        Xtr=Xtr,
        ytr=ytr,
        Xte=Xte,
        yte=yte,
        reg=args.reg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        alpha=args.naggs_alpha,
        mu=args.naggs_mu,
        gamma0=args.naggs_gamma0,
        run_dir=run_dir,
    )
    ironfi_alpha = args.ironfi_alpha
    if args.tune_ironfi:
        grid = args.ironfi_alpha_grid or [0.5, 1, 2, 5, 10, 20, 50]
        best_alpha = None
        best_acc = -1.0
        for a in grid:
            curves_tmp = run_ironfi(
                Xtr=Xtr,
                ytr=ytr,
                Xte=Xte,
                yte=yte,
                reg=args.reg,
                epochs=args.tune_epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                alpha=a,
                mu=args.ironfi_mu,
                gamma0=args.ironfi_gamma0,
                inner_tol=args.ironfi_inner_tol,
                inner_newton=args.ironfi_inner_newton,
                cg_tol=args.ironfi_cg_tol,
                cg_max_it=args.ironfi_cg_max_it,
                run_dir=run_dir,
            )
            acc = float(curves_tmp["test_acc"][-1])
            if acc > best_acc:
                best_acc = acc
                best_alpha = a
        ironfi_alpha = float(best_alpha)
        print(f"[tune] selected IRON-FI alpha={ironfi_alpha:g} (tune_epochs={args.tune_epochs}, acc={best_acc:.4f})")

    curves_ironfi = run_ironfi(
        Xtr=Xtr,
        ytr=ytr,
        Xte=Xte,
        yte=yte,
        reg=args.reg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        alpha=ironfi_alpha,
        alpha2=args.ironfi_alpha2,
        alpha2_start_epoch=args.ironfi_alpha2_start_epoch,
        mu=args.ironfi_mu,
        gamma0=args.ironfi_gamma0,
        inner_tol=args.ironfi_inner_tol,
        inner_newton=args.ironfi_inner_newton,
        cg_tol=args.ironfi_cg_tol,
        cg_max_it=args.ironfi_cg_max_it,
        run_dir=run_dir,
    )

    # Plots
    e = np.arange(args.epochs)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.4), dpi=150)
    ax.plot(e, curves_adamw["train_loss"], label=f"AdamW lr={args.adamw_lr:g}")
    ax.plot(e, curves_naggs["train_loss"], label=f"NAG-GS alpha={args.naggs_alpha:g}")
    ax.plot(e, curves_ironfi["train_loss"], label=f"IRON-FI alpha={ironfi_alpha:g}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    tag = f"_{args.fig_tag}" if args.fig_tag else ""
    fig.savefig(f"figs/mnist_softmax_train_loss{tag}.pdf")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.4), dpi=150)
    ax.plot(e, curves_adamw["test_acc"], label=f"AdamW lr={args.adamw_lr:g}")
    ax.plot(e, curves_naggs["test_acc"], label=f"NAG-GS alpha={args.naggs_alpha:g}")
    ax.plot(e, curves_ironfi["test_acc"], label=f"IRON-FI alpha={ironfi_alpha:g}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.87, 0.94)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"figs/mnist_softmax_test_acc{tag}.pdf")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.4), dpi=150)
    ax.plot(e, curves_ironfi["inner_newton"], "o-", label="mean Newton iters/epoch")
    ax.plot(e, curves_ironfi["inner_res"], "o-", label="mean residual/epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("inner stats (IRON-FI)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"figs/mnist_softmax_ironfi_inner_stats{tag}.pdf")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

