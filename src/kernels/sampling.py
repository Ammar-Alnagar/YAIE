"""
Sampling Kernel implementation
Implements Top-P, Top-K, and Temperature sampling.
"""

import torch

class SamplingKernel:
    """
    Kernel for sampling from probability distributions.
    """

    def __init__(self):
        pass

    def sample(self, logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0, top_k: int = -1) -> torch.Tensor:
        """
        Sample token IDs from logits using temperature, top_p, and top_k.

        Args:
            logits: Input logits [batch_size, vocab_size]
            temperature: Sampling temperature
            top_p: Cumulative probability threshold for nucleus sampling
            top_k: Top-k tokens to keep

        Returns:
            Selected token IDs [batch_size]
        """
        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Get probabilities
        probs = torch.softmax(logits, dim=-1)

        batch_size, vocab_size = probs.shape

        # Apply Top-K filtering if specified
        if top_k > 0:
            top_k = min(top_k, vocab_size)  # Don't exceed vocab size
            # Get the top-k values and indices
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)

            # Create a new probability distribution with only top-k values
            new_probs = torch.zeros_like(probs)
            new_probs.scatter_(1, top_k_indices, top_k_probs)
            # Re-normalize
            new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)
            probs = new_probs

        # Apply Top-P (nucleus) filtering if specified
        if 0 < top_p < 1.0:
            # Sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

            # Calculate cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Create mask to keep only the top probabilities that sum to top_p
            mask = cumulative_probs <= top_p

            # Ensure at least one token is kept (the first one)
            if mask.shape[-1] > 0:
                mask[..., 0] = True  # Always keep at least the first (highest) probability

            # Only keep probabilities where mask is True
            filtered_probs = torch.zeros_like(probs)
            # Use gather to place the filtered probabilities back in their original positions
            filtered_probs.scatter_(1, sorted_indices, mask.float() * sorted_probs)

            # Re-normalize
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
            probs = filtered_probs

        # Sample from the final probability distribution
        sampled_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return sampled_ids
