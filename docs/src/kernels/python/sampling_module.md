# Sampling (`kernels/sampling.py`)

## 1. Concept: From Logits to Tokens

The model outputs **logits**: a vector of size `[VOCAB_SIZE]` (e.g., 50,000) containing raw scores for the next token.
We need to pick **one** token ID.

- **Greedy**: Just pick `argmax()`. Boring, repetitive.
- **Sampling**: Pick randomly based on probability. Creative!

We control the randomness with **Temperature**, **Top-P** (Nucleus), and **Top-K**.

---

## 2. Implementation Guide

Open `src/kernels/sampling.py`.

### Step 1: Temperature Scaling

**Temperature** ($T$) controls confidence.

- $T < 1$: Makes peakier (more confident).
- $T > 1$: Makes flatter (more random).

**Algorithm**:

```python
logits = logits / temperature
```

- **Watch out**: If $T$ is very close to 0, just do `argmax` to avoid division by zero!

---

### Step 2: Softmax

Convert logits to probabilities (0.0 to 1.0).

```python
probs = torch.softmax(logits, dim=-1)
```

---

### Step 3: Top-K Filtering

Keep only the $K$ most likely tokens. Zero out the rest.

**Algorithm**:

1.  Find the value of the $K$-th highest score.
2.  Mask (set to $-\infty$) anything below that value in `logits` (or 0 in `probs`).
3.  Re-normalize probabilities.

---

### Step 4: Top-P (Nucleus) Filtering (The Tricky One)

Keep the smallest set of tokens whose cumulative probability adds up to $P$ (e.g., 0.9). This dynamically truncates the long tail of "nonsense" words.

**Algorithm**:

1.  Sort probabilities in descending order: `sorted_probs, sorted_indices = torch.sort(probs, descending=True)`.
2.  Compute cumulative sum: `cumulative_probs = torch.cumsum(sorted_probs, dim=-1)`.
3.  Find cut-off: Mask where `cumulative_probs > top_p`.
    - _Tip_: You want to include the _first_ token that crosses the threshold. So shift the mask right by one.
4.  Scatter the mask back to the original ordering.
5.  Re-normalize.

---

### Step 5: The Final Selection

Once you have your clean probability distribution:

```python
next_token = torch.multinomial(probs, num_samples=1)
```

**Your Turn**: Implement `sample` in `SamplingKernel`. Start simple (just Temperature) and verify, then add Top-P.
