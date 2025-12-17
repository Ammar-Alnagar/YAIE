# Sampling Module (`sampling.py`)

The `sampling.py` module provides the `SamplingKernel` class, which encapsulates various strategies for sampling tokens from a model's output probability distribution. These strategies (Temperature, Top-P, Top-K) are crucial for controlling the diversity and quality of generated text in large language models.

## `SamplingKernel` Class

This class is responsible for applying different sampling techniques to raw model logits to select the next token in a sequence.

### `__init__(self)`

The constructor for the `SamplingKernel` class. It doesn't take any specific parameters as the sampling logic is primarily handled within the `sample` method.

### `sample(self, logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0, top_k: int = -1) -> torch.Tensor`

This method takes raw logits from a language model and applies a combination of temperature scaling, Top-K filtering, and Top-P (nucleus) sampling to select the next token(s).

*   **Parameters**:
    *   `logits` (torch.Tensor):
        The raw, unnormalized scores output by the model for each token in the vocabulary. Expected shape is `[batch_size, vocab_size]`.
    *   `temperature` (float, default: `1.0`):
        A parameter that controls the randomness of the sampling process.
        *   A `temperature` of `1.0` means no change to the logits.
        *   Values greater than `1.0` (e.g., `1.5`) make the distribution flatter, increasing the probability of less likely tokens and leading to more diverse/random outputs.
        *   Values less than `1.0` (e.g., `0.7`) make the distribution sharper, concentrating probability mass on more likely tokens and leading to more deterministic outputs.
    *   `top_p` (float, default: `1.0`):
        The cumulative probability threshold for Nucleus (Top-P) sampling. If `0 < top_p < 1.0`, only the smallest set of most probable tokens whose cumulative probability exceeds `top_p` are considered. This helps in dynamically adjusting the vocabulary size based on the probability distribution.
    *   `top_k` (int, default: `-1`):
        The number of top most probable tokens to consider for sampling. If `top_k > 0`, only the `top_k` most likely tokens are kept. This effectively reduces the vocabulary size to a fixed number of most probable tokens. Set to `-1` to disable Top-K filtering.

*   **Process**:
    1.  **Temperature Scaling**: The `logits` are divided by the `temperature` to either flatten or sharpen the probability distribution.
    2.  **Probability Conversion**: The scaled `logits` are converted into a probability distribution using the `torch.softmax` function.
    3.  **Top-K Filtering**: If `top_k` is specified and greater than 0, the method identifies the `top_k` most probable tokens and sets the probabilities of all other tokens to zero. The distribution is then re-normalized.
    4.  **Top-P (Nucleus) Filtering**: If `top_p` is specified and between 0 and 1, the method sorts the probabilities in descending order. It then calculates the cumulative probabilities and keeps only the tokens that fall within the `top_p` cumulative probability mass. All other token probabilities are set to zero, and the distribution is re-normalized. Crucially, at least one token (the highest probability one) is always kept to prevent an empty vocabulary.
    5.  **Sampling**: Finally, `torch.multinomial` is used to draw a single token ID from the (potentially filtered and re-normalized) probability distribution.

*   **Returns**:
    `torch.Tensor`: A tensor containing the sampled token ID(s). The shape is `[batch_size]`, representing one token sampled for each sequence in the batch.
