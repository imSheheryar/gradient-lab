"""gradient-lab: a small autodiff engine built to be instrumented."""

from .gradcheck import check, numerical_grad, relative_error
from .nn import Dropout, GELU, Identity, Linear, Module, ReLU, Sequential, Sigmoid, Tanh
from .ops import layernorm, softmax, softmax_cross_entropy
from .optim import SGD, Adam, AdamW
from .tensor import Tensor

__all__ = [
    "Tensor",
    "check",
    "numerical_grad",
    "relative_error",
    "layernorm",
    "softmax",
    "softmax_cross_entropy",
    "Module",
    "Linear",
    "Dropout",
    "Sequential",
    "Tanh",
    "ReLU",
    "GELU",
    "Sigmoid",
    "Identity",
    "SGD",
    "Adam",
    "AdamW",
]
