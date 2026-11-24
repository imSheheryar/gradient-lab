# gradient-lab

A reverse-mode autodiff engine written from scratch in NumPy, built as an instrument rather than a demo.

The question this repository answers: **how deep can a network get before the gradient dies, and which single intervention buys the most depth?** The engine is the measuring device. Everything below exists so that the measurement can be trusted.

Status: session 2 of 4 — the engine and the library layer are complete and verified against PyTorch.

## Verified gradients

Every operation is checked against central finite differences at float64 (`eps=1e-6`). These are measured numbers, not assertions:

```
operation                   max rel. error   grad norm
------------------------------------------------------
add (broadcast)                   2.40e-10       22.89
multiply                          1.71e-10        4.53
matmul                            8.00e-11        7.85
relu                              1.48e-10        4.00
tanh                              4.65e-11        3.08
sigmoid                           8.26e-10        0.97
gelu                              2.06e-10        3.11
exp                               4.32e-10        7.00
log                               3.43e-11        4.68
sum(axis=0)                       1.56e-10       20.27
mean(axis=1)                      8.05e-11        0.94
max(axis=1)                       6.52e-11        2.24
reshape                           2.02e-10        7.59
transpose                         1.12e-10       15.01
softmax                           5.89e-10        0.30
softmax_cross_entropy             7.55e-10        0.36
layernorm                         5.28e-10       33.89
------------------------------------------------------
worst case: 8.26e-10   tolerance: 1.0e-06   PASS
```

The worst case sits three orders of magnitude inside the tolerance, which is what float64 arithmetic on small tensors should give you. What happens to that floor at float32 is the precision study in session 3.

## Parity with PyTorch

Finite differences tell you the gradient is self-consistent. They do not tell you it is the *right* gradient. So every layer and optimiser is built a second time in PyTorch, fed identical parameters, and compared:

```
component                         max abs diff
----------------------------------------------
linear forward                        0.00e+00
linear grad wrt input                 0.00e+00
linear grad wrt weight                0.00e+00
linear grad wrt bias                  0.00e+00
layernorm forward                     8.88e-16
layernorm grad wrt input              1.42e-14
layernorm grad wrt gamma              1.78e-15
SGD + Nesterov, 5 steps               8.88e-16
Adam, 5 steps                         2.78e-17
AdamW, 5 steps                        0.00e+00
----------------------------------------------
worst deviation: 1.42e-14   tolerance 1e-12   PASS
```

Exact zeros are not suspicious: both implementations call the same BLAS routine on the same doubles. The 1e-14 in LayerNorm is the interesting one — it is the reduction order of the mean, and it is exactly the size you would predict from float64 rounding over four elements.

## It learns

XOR is not linearly separable, so a two-layer network can only solve it if gradients flow through both hidden layers.

```
step    0   loss 1.1839   accuracy 0.515
step  150   loss 0.0214   accuracy 0.996
step  600   loss 0.0128   accuracy 0.998
```

![XOR decision boundary](figures/xor_decision_boundary.png)

## It writes

A character-level language model: eight characters of context, one-hot encoded, through a 160-unit tanh layer, predicting the next character. No embeddings, no attention — the point is to put the engine on a real workload.

93,825 parameters, 6,000 Adam steps, 3.1 ms per step on one CPU core. Cross-entropy falls from 4.24 nats (uniform guessing over 65 characters) to 1.93, a perplexity of about 7.

![Character-level language model](figures/char_lm.png)

At step 0 the output is noise. At 600 it has learned word boundaries and syllable structure. At 6,000 it produces sentences that are grammatically shaped, semantically empty, and unmistakably Shakespeare-adjacent — which is exactly what an eight-character context window can buy you.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m pytest -q              # 41 tests
.venv/bin/python -m scripts.verify         # gradient table
.venv/bin/python -m scripts.verify_parity  # PyTorch table
.venv/bin/python -m scripts.smoke_train    # XOR
.venv/bin/python -m scripts.char_lm        # language model
```

## Three decisions worth knowing about

**Iterative topological sort.** The backward pass walks the graph with an explicit stack, not recursion. A 600-layer chain exceeds Python's recursion limit, and deep chains are the entire subject of this project. There is a test for exactly this.

**Gradients accumulate with `+=`.** This is what makes fan-out correct: a tensor consumed by two operations receives the sum of both contributions. It also means `zero_grad()` is mandatory between steps, as in any real framework.

**Broadcasting is undone explicitly.** When `(B, D)` and `(1, D)` are added, the gradient flowing back to the `(1, D)` operand has to be summed over the broadcast axis. `_unbroadcast` does this, and there are shape tests for it — this is the single most common silent bug in hand-written autodiff.

## What is deliberately missing

No attention, no transformer, no C extension. Those are on the cut list, and the reasoning is in `docs/scope.md`: they add engineering surface without adding to the claim.

## Roadmap

- [x] Session 1 — Tensor, DAG, topological sort, fused softmax-CE, LayerNorm, gradcheck
- [x] Session 2 — Linear, Dropout, initialisation schemes, SGD/momentum/Nesterov/Adam/AdamW, PyTorch parity, character-level language model
- [ ] Session 3 — the depth-horizon experiment and the precision study
- [ ] Session 4 — figures, probe classifier, write-up
