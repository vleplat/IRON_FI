import os
import time
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import numpy as np

from ironfi.ironfi_mf import ironfi_step_matrix_free
from ironfi.optimizers.adamw import AdamW
from ironfi.optimizers.nag_gs import nag_gs_step
from models.softmax import SoftmaxParams, accuracy, hvp as softmax_hvp, loss_and_grad
from utils.expio import append_csv_row


def iter_minibatches(indices: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for i in range(0, len(indices), batch_size):
        yield indices[i : i + batch_size]


@dataclass
class RunCurves:
    train_loss: np.ndarray
    test_acc: np.ndarray
    # IRON-only extras (None for other methods)
    inner_newton: Optional[np.ndarray] = None
    inner_cg: Optional[np.ndarray] = None
    inner_res: Optional[np.ndarray] = None


def run_adamw(
    *,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    reg: float,
    epochs: int,
    batch_size: int,
    seed: int,
    lr: float,
    weight_decay: float,
    run_dir: str,
    tag: str,
) -> RunCurves:
    rng = np.random.default_rng(seed)
    d = Xtr.shape[1]
    C = int(np.max(ytr)) + 1
    params = SoftmaxParams(W=np.zeros((d, C)), b=np.zeros((C,)))
    theta = params.pack()
    opt = AdamW(lr=lr, weight_decay=weight_decay)

    rows_path = os.path.join(run_dir, f"adamw_lr{lr:g}_{tag}.csv")
    header = ["epoch", "step", "train_loss", "test_acc", "elapsed_s"]
    t0 = time.time()
    step = 0

    tr_curve = np.zeros(epochs)
    te_curve = np.zeros(epochs)

    for ep in range(epochs):
        perm = rng.permutation(Xtr.shape[0])
        for bidx in iter_minibatches(perm, batch_size):
            p = SoftmaxParams.unpack(theta, d, C)
            _, g = loss_and_grad(p, Xtr[bidx], ytr[bidx], reg)
            theta = opt.step(theta, g.pack())
            step += 1

        p = SoftmaxParams.unpack(theta, d, C)
        sub = perm[: min(10000, Xtr.shape[0])]
        tr_loss, _ = loss_and_grad(p, Xtr[sub], ytr[sub], reg)
        te_acc = accuracy(p, Xte, yte)
        tr_curve[ep] = tr_loss
        te_curve[ep] = te_acc

        append_csv_row(
            rows_path,
            {"epoch": ep, "step": step, "train_loss": tr_loss, "test_acc": te_acc, "elapsed_s": time.time() - t0},
            header=header,
        )

    return RunCurves(train_loss=tr_curve, test_acc=te_curve)


def run_naggs(
    *,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    reg: float,
    epochs: int,
    batch_size: int,
    seed: int,
    alpha: float,
    mu: float,
    gamma0: float,
    run_dir: str,
    tag: str,
) -> RunCurves:
    rng = np.random.default_rng(seed)
    d = Xtr.shape[1]
    C = int(np.max(ytr)) + 1
    x = SoftmaxParams(W=np.zeros((d, C)), b=np.zeros((C,))).pack()
    v = x.copy()
    gamma = float(gamma0)

    rows_path = os.path.join(run_dir, f"naggs_alpha{alpha:g}_{tag}.csv")
    header = ["epoch", "step", "train_loss", "test_acc", "gamma", "elapsed_s"]
    t0 = time.time()
    step = 0

    tr_curve = np.zeros(epochs)
    te_curve = np.zeros(epochs)

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
        sub = perm[: min(10000, Xtr.shape[0])]
        tr_loss, _ = loss_and_grad(p, Xtr[sub], ytr[sub], reg)
        te_acc = accuracy(p, Xte, yte)
        tr_curve[ep] = tr_loss
        te_curve[ep] = te_acc

        append_csv_row(
            rows_path,
            {"epoch": ep, "step": step, "train_loss": tr_loss, "test_acc": te_acc, "gamma": gamma, "elapsed_s": time.time() - t0},
            header=header,
        )

    return RunCurves(train_loss=tr_curve, test_acc=te_curve)


def run_ironfi(
    *,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    reg: float,
    epochs: int,
    batch_size: int,
    seed: int,
    alpha: float,
    alpha2: float | None,
    alpha2_start_epoch: int | None,
    mu: float,
    gamma0: float,
    inner_tol: float,
    inner_newton: int,
    cg_tol: float,
    cg_max_it: int,
    run_dir: str,
    tag: str,
) -> RunCurves:
    rng = np.random.default_rng(seed)
    d = Xtr.shape[1]
    C = int(np.max(ytr)) + 1
    x = SoftmaxParams(W=np.zeros((d, C)), b=np.zeros((C,))).pack()
    v = x.copy()
    gamma = float(gamma0)

    if alpha2 is not None and alpha2_start_epoch is not None:
        rows_path = os.path.join(run_dir, f"ironfi_alpha{alpha:g}_to_{alpha2:g}_at{alpha2_start_epoch}_{tag}.csv")
    else:
        rows_path = os.path.join(run_dir, f"ironfi_alpha{alpha:g}_{tag}.csv")

    header = ["epoch", "step", "train_loss", "test_acc", "gamma", "inner_newton", "inner_cg", "inner_res", "elapsed_s"]
    t0 = time.time()
    step = 0

    tr_curve = np.zeros(epochs)
    te_curve = np.zeros(epochs)
    in_curve = np.zeros(epochs)
    icg_curve = np.zeros(epochs)
    ir_curve = np.zeros(epochs)

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
                sigma_center=0.0,
                update_gamma_flag=True,
            )
            step += 1

            inner = info["inner"]
            inner_newton_hist.append(inner.n_newton)
            inner_cg_hist.append(inner.n_cg_total)
            inner_res_hist.append(inner.final_residual_norm)

        p = SoftmaxParams.unpack(x, d, C)
        sub = perm[: min(10000, Xtr.shape[0])]
        tr_loss, _ = loss_and_grad(p, Xtr[sub], ytr[sub], reg)
        te_acc = accuracy(p, Xte, yte)

        tr_curve[ep] = tr_loss
        te_curve[ep] = te_acc
        in_curve[ep] = float(np.mean(inner_newton_hist))
        icg_curve[ep] = float(np.mean(inner_cg_hist))
        ir_curve[ep] = float(np.mean(inner_res_hist))

        append_csv_row(
            rows_path,
            {
                "epoch": ep,
                "step": step,
                "train_loss": tr_loss,
                "test_acc": te_acc,
                "gamma": gamma,
                "inner_newton": in_curve[ep],
                "inner_cg": icg_curve[ep],
                "inner_res": ir_curve[ep],
                "elapsed_s": time.time() - t0,
            },
            header=header,
        )

    return RunCurves(
        train_loss=tr_curve,
        test_acc=te_curve,
        inner_newton=in_curve,
        inner_cg=icg_curve,
        inner_res=ir_curve,
    )

