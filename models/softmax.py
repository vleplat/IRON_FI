from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class SoftmaxParams:
    W: np.ndarray  # (d, C)
    b: np.ndarray  # (C,)

    def pack(self) -> np.ndarray:
        """
        Pack parameters into a flat vector.

        Returns
        -------
        theta : np.ndarray
            Flattened parameter vector of shape (d*C + C,).
        """
        return np.concatenate([self.W.ravel(), self.b.ravel()])

    @staticmethod
    def unpack(theta: np.ndarray, d: int, C: int) -> "SoftmaxParams":
        """
        Unpack a flat vector into (W,b).

        Parameters
        ----------
        theta : np.ndarray
            Flattened parameter vector of shape (d*C + C,).
        d : int
            Feature dimension.
        C : int
            Number of classes.

        Returns
        -------
        params : SoftmaxParams
            Structured parameters.
        """
        w = theta[: d * C].reshape(d, C)
        b = theta[d * C : d * C + C].reshape(C)
        return SoftmaxParams(W=w, b=b)


def _matmul_checked(A: np.ndarray, B: np.ndarray, name: str) -> np.ndarray:
    # Some BLAS backends can emit spurious FP warnings; we suppress and explicitly check finiteness.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        out = A @ B
    if not np.all(np.isfinite(out)):
        raise FloatingPointError(f"Non-finite values encountered in {name}.")
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def loss_and_grad(
    params: SoftmaxParams,
    X: np.ndarray,
    y: np.ndarray,
    reg: float,
) -> Tuple[float, SoftmaxParams]:
    """
    Cross-entropy loss + L2 on W (not on b), averaged over batch.
    y in {0,...,C-1}.

    Parameters
    ----------
    params : SoftmaxParams
        Model parameters.
    X : np.ndarray
        Batch features of shape (m, d).
    y : np.ndarray
        Batch labels of shape (m,), values in {0,...,C-1}.
    reg : float
        L2 regularization strength on W.

    Returns
    -------
    loss : float
        Regularized cross-entropy loss (mean over batch).
    grad : SoftmaxParams
        Gradient w.r.t. (W,b), same shapes as params.
    """
    m = X.shape[0]
    logits = _matmul_checked(X, params.W, "X @ W (logits)") + params.b[None, :]
    P = _softmax(logits)

    # loss
    logp = np.log(P[np.arange(m), y] + 1e-12)
    ce = -float(np.mean(logp))
    loss = ce + 0.5 * reg * float(np.sum(params.W * params.W))

    # grad
    G = P.copy()
    G[np.arange(m), y] -= 1.0
    G /= m
    gW = _matmul_checked(X.T, G, "X.T @ G (grad_W)") + reg * params.W
    gb = np.sum(G, axis=0)
    return loss, SoftmaxParams(W=gW, b=gb)


def hvp(
    params: SoftmaxParams,
    X: np.ndarray,
    y: np.ndarray,
    reg: float,
    vec: SoftmaxParams,
) -> SoftmaxParams:
    """
    Hessian-vector product for softmax regression + L2(W).
    Uses the identity:
      (diag(p)-p p^T) t = p ⊙ (t - <p,t>)

    Parameters
    ----------
    params : SoftmaxParams
        Current parameters (defines the Hessian point).
    X : np.ndarray
        Batch features (m, d).
    y : np.ndarray
        Batch labels (m,).
    reg : float
        L2 regularization on W.
    vec : SoftmaxParams
        Vector to multiply by the Hessian.

    Returns
    -------
    hv : SoftmaxParams
        Hessian(params) @ vec.
    """
    m = X.shape[0]
    logits = _matmul_checked(X, params.W, "X @ W (hvp logits)") + params.b[None, :]
    P = _softmax(logits)  # (m,C)

    T = _matmul_checked(X, vec.W, "X @ Vw (hvp)") + vec.b[None, :]  # (m,C)
    dot = np.sum(P * T, axis=1, keepdims=True)  # (m,1)
    Hv_logits = P * (T - dot)  # (m,C)
    hW = _matmul_checked(X.T, (Hv_logits / m), "X.T @ Hv_logits (hvp_W)") + reg * vec.W
    hb = np.sum(Hv_logits / m, axis=0)
    return SoftmaxParams(W=hW, b=hb)


def accuracy(params: SoftmaxParams, X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Parameters
    ----------
    params : SoftmaxParams
        Model parameters.
    X : np.ndarray
        Features (N, d).
    y : np.ndarray
        Labels (N,).

    Returns
    -------
    acc : float
        Fraction of correct predictions.
    """
    logits = _matmul_checked(X, params.W, "X @ W (acc logits)") + params.b[None, :]
    pred = np.argmax(logits, axis=1)
    return float(np.mean(pred == y))

