"""A character-level language model trained entirely on this engine.

Context window of N characters, one-hot encoded, through a small MLP, predicting
the next character. No embeddings, no attention -- the point is to put the engine
on a real workload and show that it learns something that reads like language.

Run:  .venv/bin/python -m scripts.char_lm --steps 2000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from engine import Adam, Linear, Sequential, Tanh, Tensor
from engine.ops import softmax_cross_entropy

ROOT = Path(__file__).resolve().parent.parent


def load_corpus() -> str:
    path = ROOT / "data" / "tinyshakespeare.txt"
    if not path.exists():
        raise SystemExit(f"corpus missing at {path}")
    return path.read_text(encoding="utf-8")


def build_vocab(text: str):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    return chars, stoi


def make_windows(encoded: np.ndarray, context: int):
    """All (context, next-character) pairs, as index arrays."""
    n = len(encoded) - context
    idx = np.arange(n)[:, None] + np.arange(context)[None, :]
    return encoded[idx], encoded[context : context + n]


def one_hot(contexts: np.ndarray, vocab_size: int) -> np.ndarray:
    eye = np.eye(vocab_size)
    return eye[contexts].reshape(len(contexts), -1)


def sample(model, chars, stoi, seed: str, length: int, context: int,
           temperature: float = 0.8, rng=None):
    rng = rng or np.random.default_rng()
    out = list(seed)
    window = [stoi[c] for c in seed[-context:]]
    for _ in range(length):
        x = Tensor(one_hot(np.array([window]), len(chars)))
        logits = model(x).data[0]
        logits = logits / temperature
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        nxt = int(rng.choice(len(chars), p=probs))
        out.append(chars[nxt])
        window = (window + [nxt])[1:]
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    text = load_corpus()
    chars, stoi = build_vocab(text)
    vocab = len(chars)
    encoded = np.array([stoi[c] for c in text], dtype=np.int64)
    contexts, targets = make_windows(encoded, args.context)

    model = Sequential(
        Linear(args.context * vocab, args.hidden, init="he_normal", rng=rng),
        Tanh(),
        Linear(args.hidden, vocab, init="xavier_normal", rng=rng),
    )
    opt = Adam(model.parameters(), lr=args.lr)

    print(f"corpus {len(text):,} chars | vocab {vocab} | windows {len(contexts):,}")
    print(f"model {args.context * vocab} -> {args.hidden} -> {vocab}\n")

    history = []
    start = time.time()
    for step in range(args.steps + 1):
        batch_idx = rng.integers(0, len(contexts), args.batch)
        x = Tensor(one_hot(contexts[batch_idx], vocab))
        loss = softmax_cross_entropy(model(x), targets[batch_idx])

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            history.append((step, float(loss.data)))
            print(f"step {step:>5}   loss {float(loss.data):.4f}   "
                  f"{(time.time() - start):.1f}s")

    elapsed = time.time() - start
    print(f"\ntrained {args.steps} steps in {elapsed:.1f}s "
          f"({elapsed / max(args.steps, 1) * 1000:.0f} ms/step)\n")

    seed_text = "First Citizen:\n"
    for label, temperature in (("greedy-ish (T=0.5)", 0.5), ("sampled (T=0.8)", 0.8)):
        print(f"--- {label} ---")
        print(sample(model, chars, stoi, seed_text, 240, args.context, temperature, rng))
        print()

    np.save(ROOT / "figures" / "char_lm_loss.npy", np.array(history))
    print("loss history saved to figures/char_lm_loss.npy")


if __name__ == "__main__":
    main()
