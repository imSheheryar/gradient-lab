"""Layers, built only out of operations the engine already knows how to differentiate."""

from __future__ import annotations

import numpy as np

from . import init as init_module
from .tensor import Tensor


class Module:
    """Minimal module protocol: parameters, train/eval, call."""

    training: bool = True

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - overridden
        raise NotImplementedError

    def parameters(self) -> list[Tensor]:
        return []

    def children(self) -> list["Module"]:
        return []

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for child in self.children():
            child.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)


class Linear(Module):
    """y = x @ W + b, with the initialisation scheme as an explicit choice."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "he_normal",
        rng: np.random.Generator | None = None,
    ) -> None:
        rng = rng or np.random.default_rng()
        scheme = init_module.get(init)
        self.weight = Tensor(scheme((in_features, out_features), rng))
        self.bias = Tensor(np.zeros(out_features)) if bias else None
        self.in_features = in_features
        self.out_features = out_features
        self.init = init

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        return out + self.bias if self.bias is not None else out

    def parameters(self) -> list[Tensor]:
        return [self.weight] if self.bias is None else [self.weight, self.bias]


class Dropout(Module):
    """Inverted dropout: scale by 1/(1-p) at train time, identity at eval time."""

    def __init__(self, p: float = 0.1, rng: np.random.Generator | None = None) -> None:
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.p = p
        self.rng = rng or np.random.default_rng()

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * Tensor(keep, requires_grad=False)


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        self.modules = list(modules)

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x

    def children(self) -> list[Module]:
        return self.modules

    def parameters(self) -> list[Tensor]:
        return [p for m in self.modules for p in m.parameters()]


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x
