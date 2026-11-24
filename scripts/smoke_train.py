"""Train a 2-layer MLP on XOR using nothing but the engine.

This is the smallest end-to-end proof that the backward pass is correct: XOR is
not linearly separable, so the network can only solve it if gradients flow
through both hidden layers.

Run:  .venv/bin/python -m scripts.smoke_train
"""

from __future__ import annotations

import numpy as np

from engine import Tensor
from engine.ops import softmax_cross_entropy

rng = np.random.default_rng(0)

# XOR: four clusters, labels determined by sign agreement
n_per = 120
centres = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
labels = np.array([0, 1, 1, 0])
X = np.concatenate([c + 0.35 * rng.standard_normal((n_per, 2)) for c in centres])
y = np.repeat(labels, n_per)

x = Tensor(X)
targets = y

w1 = Tensor(rng.standard_normal((2, 16)) * 0.8)
b1 = Tensor(np.zeros(16))
w2 = Tensor(rng.standard_normal((16, 16)) * 0.8)
b2 = Tensor(np.zeros(16))
w3 = Tensor(rng.standard_normal((16, 2)) * 0.8)
b3 = Tensor(np.zeros(2))
params = [w1, b1, w2, b2, w3, b3]

lr = 0.15
for step in range(601):
    h1 = (x @ w1 + b1).tanh()
    h2 = (h1 @ w2 + b2).tanh()
    logits = h2 @ w3 + b3
    loss = softmax_cross_entropy(logits, targets)

    for p in params:
        p.zero_grad()
    loss.backward()
    for p in params:
        p.data -= lr * p.grad

    if step % 150 == 0:
        preds = logits.data.argmax(axis=1)
        acc = (preds == targets).mean()
        print(f"step {step:>4}   loss {float(loss.data):.4f}   accuracy {acc:.3f}")

preds = logits.data.argmax(axis=1)
print(f"\nfinal accuracy: {(preds == targets).mean():.3f} on {len(targets)} points")
print(f"loss: {float(loss.data):.4f}")
