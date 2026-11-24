"""Optimisers, written to match torch.optim step for step.

The parity suite compares these against PyTorch on identical parameters, which
is the only way to be sure a hand-written Adam is actually Adam and not a
plausible-looking cousin.
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Optimizer:
    def __init__(self, params: list[Tensor], lr: float) -> None:
        self.params = list(params)
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-2,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires momentum > 0")
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.buffers = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for p, buf in zip(self.params, self.buffers):
            grad = p.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            if self.momentum:
                buf *= self.momentum
                buf += grad
                direction = grad + self.momentum * buf if self.nesterov else buf
            else:
                direction = grad
            p.data -= self.lr * direction


class Adam(Optimizer):
    """Adam with coupled weight decay (the original formulation)."""

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        bias1 = 1.0 - self.beta1**self.t
        bias2 = 1.0 - self.beta2**self.t
        for p, m, v in zip(self.params, self.m, self.v):
            grad = p.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            m *= self.beta1
            m += (1.0 - self.beta1) * grad
            v *= self.beta2
            v += (1.0 - self.beta2) * grad**2
            m_hat = m / bias1
            v_hat = v / bias2
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019)."""

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay=0.0)
        self.decay = weight_decay

    def step(self) -> None:
        if self.decay:
            for p in self.params:
                p.data -= self.lr * self.decay * p.data
        super().step()
