"""gradient-lab: a small autodiff engine built to be instrumented."""

from .gradcheck import check, numerical_grad, relative_error
from .ops import layernorm, softmax, softmax_cross_entropy
from .tensor import Tensor

__all__ = [
    "Tensor",
    "check",
    "numerical_grad",
    "relative_error",
    "layernorm",
    "softmax",
    "softmax_cross_entropy",
]
