"""Core tensor with reverse-mode automatic differentiation over a dynamic DAG.

Design notes
------------
* Every operation returns a new Tensor whose `_prev` holds its parents and whose
  `_backward` closure knows how to push the upstream gradient into them.
* Gradients accumulate with `+=`, which is what makes fan-out correct: a tensor
  used twice in the graph receives the sum of both contributions.
* `backward()` runs an *iterative* depth-first search. A recursive
  implementation dies on deep graphs (a 600-layer chain exceeds Python's
  default recursion limit), and deep graphs are the whole point of this project.
* Broadcasting is handled explicitly: an op that broadcasts its inputs must
  reduce the gradient back over the broadcast axes before accumulating.
"""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray | float | int | list


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum `grad` down to `shape`, undoing numpy broadcasting."""
    if grad.shape == shape:
        return grad
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


def _coerce(value: ArrayLike, dtype: np.dtype) -> "Tensor":
    """Turn a constant into a Tensor of the same dtype as the other operand.

    Without this, `x * 2.0` on a float32 tensor would silently promote the
    whole expression to float64 and the precision study would be measuring
    the wrong thing.
    """
    if isinstance(value, Tensor):
        if value.data.dtype == dtype:
            return value
        return Tensor(value.data.astype(dtype), requires_grad=value.requires_grad)
    return Tensor(np.asarray(value, dtype=dtype))


class Tensor:
    """A dense array that remembers how it was computed."""

    __array_priority__ = 100

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = True,
        _children: tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        array = np.asarray(data)
        if not np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float64)
        self.data = array
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._prev = _children
        self._backward = lambda: None
        self._op = _op

    # ------------------------------------------------------------------ utils
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    def item(self) -> float:
        return float(self.data)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, op={self._op or 'leaf'})"

    # -------------------------------------------------------------- backward
    def backward(self, grad: np.ndarray | None = None) -> None:
        """Accumulate d(self)/d(every upstream tensor) into `.grad`."""
        if grad is None:
            if self.data.size != 1:
                raise ValueError(
                    "backward() without an explicit gradient requires a scalar output"
                )
            self.grad = np.ones_like(self.data)
        else:
            self.grad = np.asarray(grad, dtype=np.float64)

        order: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._prev:
                if id(parent) not in visited:
                    stack.append((parent, False))

        for node in reversed(order):
            node._backward()

    # ----------------------------------------------------------- arithmetic
    def __add__(self, other: ArrayLike) -> "Tensor":
        other = _coerce(other, self.data.dtype)
        out = Tensor(self.data + other.data, _children=(self, other), _op="+")

        def _backward() -> None:
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self + other

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other: ArrayLike) -> "Tensor":
        return self + (-_coerce(other, self.data.dtype))

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return _coerce(other, self.data.dtype) + (-self)

    def __mul__(self, other: ArrayLike) -> "Tensor":
        other = _coerce(other, self.data.dtype)
        out = Tensor(self.data * other.data, _children=(self, other), _op="*")

        def _backward() -> None:
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self * other

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        other = _coerce(other, self.data.dtype)
        out = Tensor(self.data / other.data, _children=(self, other), _op="/")

        def _backward() -> None:
            self.grad += _unbroadcast(out.grad / other.data, self.data.shape)
            other.grad += _unbroadcast(
                -out.grad * self.data / (other.data**2), other.data.shape
            )

        out._backward = _backward
        return out

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return _coerce(other, self.data.dtype) / self

    def __pow__(self, exponent: float) -> "Tensor":
        if not isinstance(exponent, (int, float)):
            raise TypeError("only scalar exponents are supported")
        out = Tensor(self.data**exponent, _children=(self,), _op=f"**{exponent}")

        def _backward() -> None:
            self.grad += out.grad * exponent * self.data ** (exponent - 1)

        out._backward = _backward
        return out

    def __matmul__(self, other: ArrayLike) -> "Tensor":
        other = _coerce(other, self.data.dtype)
        if self.ndim < 2 or other.ndim < 2:
            raise ValueError("matmul requires tensors with ndim >= 2")
        out = Tensor(self.data @ other.data, _children=(self, other), _op="@")

        def _backward() -> None:
            ga = out.grad @ np.swapaxes(other.data, -1, -2)
            gb = np.swapaxes(self.data, -1, -2) @ out.grad
            self.grad += _unbroadcast(ga, self.data.shape)
            other.grad += _unbroadcast(gb, other.data.shape)

        out._backward = _backward
        return out

    def __rmatmul__(self, other: ArrayLike) -> "Tensor":
        return _coerce(other, self.data.dtype) @ self

    # ------------------------------------------------------------- reductions
    def sum(self, axis: int | None = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), _children=(self,), _op="sum")

        def _backward() -> None:
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis: int | None = None, keepdims: bool = False) -> "Tensor":
        count = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / count

    def max(self, axis: int | None = None, keepdims: bool = False) -> "Tensor":
        """Maximum; gradient routes to the first maximal element only."""
        data = self.data.max(axis=axis, keepdims=keepdims)
        out = Tensor(data, _children=(self,), _op="max")

        def _backward() -> None:
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            mask = self.data == self.data.max(axis=axis, keepdims=True)
            # split the gradient across ties so the total is preserved
            share = mask / mask.sum(axis=axis, keepdims=True)
            self.grad += np.broadcast_to(grad, self.data.shape) * share

        out._backward = _backward
        return out

    # --------------------------------------------------------------- shaping
    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(self.data.reshape(*shape), _children=(self,), _op="reshape")

        def _backward() -> None:
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> "Tensor":
        axes = axes or tuple(reversed(range(self.ndim)))
        out = Tensor(self.data.transpose(axes), _children=(self,), _op="transpose")

        def _backward() -> None:
            self.grad += out.grad.transpose(np.argsort(axes))

        out._backward = _backward
        return out

    # ----------------------------------------------------------- activations
    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0.0), _children=(self,), _op="relu")

        def _backward() -> None:
            self.grad += out.grad * (self.data > 0.0)

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        data = np.tanh(self.data)
        out = Tensor(data, _children=(self,), _op="tanh")

        def _backward() -> None:
            self.grad += out.grad * (1.0 - data**2)

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        # numerically stable in both tails
        data = np.where(
            self.data >= 0,
            1.0 / (1.0 + np.exp(-self.data)),
            np.exp(self.data) / (1.0 + np.exp(self.data)),
        )
        out = Tensor(data, _children=(self,), _op="sigmoid")

        def _backward() -> None:
            self.grad += out.grad * data * (1.0 - data)

        out._backward = _backward
        return out

    def gelu(self) -> "Tensor":
        """tanh approximation of GELU, as used in GPT-2."""
        c = np.sqrt(2.0 / np.pi)
        inner = c * (self.data + 0.044715 * self.data**3)
        t = np.tanh(inner)
        data = 0.5 * self.data * (1.0 + t)
        out = Tensor(data, _children=(self,), _op="gelu")

        def _backward() -> None:
            dinner = c * (1.0 + 3 * 0.044715 * self.data**2)
            self.grad += out.grad * (0.5 * (1.0 + t) + 0.5 * self.data * (1.0 - t**2) * dinner)

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        data = np.exp(self.data)
        out = Tensor(data, _children=(self,), _op="exp")

        def _backward() -> None:
            self.grad += out.grad * data

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), _children=(self,), _op="log")

        def _backward() -> None:
            self.grad += out.grad / self.data

        out._backward = _backward
        return out
