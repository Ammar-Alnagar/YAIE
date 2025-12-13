# Sampling (`kernels/sampling.py`)

## Concept

Convert the model's raw output logits (probabilities) into a single token ID.

## Implementation Goal

Implement `SamplingKernel.sample`:

### 1. Temperature

- `logits = logits / temperature`
- Higher temp = flatter distribution (more random).
- Lower temp = sharper distribution (more deterministic).

### 2. Top-P (Nucleus)

- Sort probabilities descending.
- Compute cumulative sum.
- Cut off where sum > `top_p`.
- Renormalize remaining probabilities.

### 3. Selection

- `torch.multinomial(probs, 1)`
- Return the selected token ID.
