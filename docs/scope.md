# Scope: what this project is, and what it deliberately is not

## The claim

Depth does not kill gradients on its own. The interaction between activation, initialisation and normalisation does, and it has a measurable horizon: for each combination there is a depth beyond which a fixed-size network stops reaching a fixed accuracy threshold.

The deliverable is not "an autodiff engine". It is a table and a figure that answer:

1. At what depth does each configuration stop training?
2. How much width buys one layer of depth, under each intervention?
3. At what point does the finite-difference gradient check stop being informative, as a function of dtype?

## The design

A factorial grid: activation (sigmoid, tanh, ReLU, GELU) x initialisation (naive, Xavier, He) x normalisation (none, LayerNorm) x residual (on, off) x depth (2, 4, 8, 16, 32, 64) x 3 seeds. 720 runs of small networks — a few hours of CPU, no GPU. The models are deliberately tiny because the result is about optimisation geometry, not scale.

## Cut on purpose

**Multi-head attention and a transformer.** Doubles the implementation work, adds nothing to the claim, and every second repository has one. LayerNorm stays because it is a variable in the experiment.

**SwiGLU.** Same reason.

**The C / Numba kernel.** That is a performance-engineering project wearing this project's clothes. It would consume two days and produce a footnote. A `cProfile` table comparing the engine to NumPy's BLAS calls is enough.

**Memory profiling beyond one figure.** Bytes per parameter versus depth is one plot; a full allocator study is a different repository.

## Why the cut list is the point

The failure mode for this genre is a repository that implements everything and concludes nothing. Twelve features, no finding, and a reader who closes the tab. Four features and one number that nobody had published is a better trade, and it is the trade this repository makes.
