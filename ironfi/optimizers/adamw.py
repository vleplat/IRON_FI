from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AdamW:
    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0

    def __post_init__(self):
        self.t = 0
        self.m = None
        self.v = None

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)

        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad * grad)

        mhat = self.m / (1.0 - self.beta1**self.t)
        vhat = self.v / (1.0 - self.beta2**self.t)

        step = mhat / (np.sqrt(vhat) + self.eps)
        theta = theta - self.lr * step

        # decoupled weight decay
        if self.weight_decay != 0.0:
            theta = theta - self.lr * self.weight_decay * theta

        return theta

