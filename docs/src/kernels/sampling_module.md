# Sampling Kernel: Token Selection Implementation

## Overview

The sampling kernel in Mini-YAIE implements core token selection algorithms for LLM generation. It provides the essential functionality for converting model logits into actual token IDs during the inference process. The kernel implements temperature scaling, Top-K sampling, and Top-P (nucleus) sampling to give users control over generation randomness and quality.

## Core Concepts

### Token Sampling in LLMs

During LLM inference, the model outputs raw logits (unnormalized scores) for each possible token in the vocabulary. The sampling kernel transforms these logits into a probability distribution and selects the next token based on this distribution.

### Key Sampling Strategies

The kernel implements three primary sampling strategies:

1. **Temperature Scaling**: Controls the randomness of predictions
2. **Top-K Sampling**: Limits selection to the K most likely tokens
3. **Top-P (Nucleus) Sampling**: Limits selection to tokens that sum to probability P

## Architecture

### SamplingKernel Class

The core implementation is in the `SamplingKernel` class:

```python
class SamplingKernel:
    def __init__(self):
        pass  # Currently no initialization required

    def sample(self, logits: torch.Tensor, temperature: float = 1.0, 
               top_p: float = 1.0, top_k: int = -1) -> torch.Tensor:
        # Main sampling implementation
```

### Input Parameters

The sampling method accepts:

- **logits**: Raw model outputs `[batch_size, vocab_size]`
- **temperature**: Scaling factor for randomness (default: 1.0)
- **top_p**: Cumulative probability threshold (default: 1.0)
- **top_k**: Number of top tokens to consider (default: -1, meaning no limit)

## Detailed Implementation

### 1. Temperature Scaling

Temperature scaling adjusts the randomness of the output:

```python
if temperature != 1.0:
    logits = logits / temperature
```

**Effect:**
- Temperature > 1.0: Increases randomness, more diverse output
- Temperature < 1.0: Decreases randomness, more deterministic output
- Temperature = 1.0: No change, standard sampling

### 2. Probability Conversion

Convert logits to probabilities using softmax:

```python
probs = torch.softmax(logits, dim=-1)
```

This creates a proper probability distribution where all probabilities sum to 1.0.

### 3. Top-K Filtering

Limit sampling to the K most probable tokens:

```python
if top_k > 0:
    top_k = min(top_k, vocab_size)  # Don't exceed vocabulary size
    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
    
    # Create new probability distribution with only top-k values
    new_probs = torch.zeros_like(probs)
    new_probs.scatter_(1, top_k_indices, top_k_probs)
    new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)  # Re-normalize
    probs = new_probs
```

**Process:**
1. Extract top K probabilities and their indices
2. Create a new probability distribution containing only these values
3. Zero out all other probabilities
4. Re-normalize to ensure probabilities sum to 1.0

### 4. Top-P (Nucleus) Filtering

Select the smallest set of tokens whose cumulative probability exceeds P:

```python
if 0 < top_p < 1.0:
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Create mask for tokens to keep
    mask = cumulative_probs <= top_p
    if mask.shape[-1] > 0:
        mask[..., 0] = True  # Always keep at least the highest probability token
    
    # Apply mask to create filtered distribution
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(1, sorted_indices, mask.float() * sorted_probs)
    
    # Re-normalize
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    probs = filtered_probs
```

**Process:**
1. Sort tokens by probability in descending order
2. Calculate cumulative probability sum
3. Identify tokens whose cumulative probability ≤ top_p
4. Ensure at least the highest probability token is kept
5. Create filtered probability distribution
6. Re-normalize to sum to 1.0

### 5. Token Selection

Finally, sample from the final probability distribution:

```python
sampled_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
```

This performs multinomial sampling (weighted random selection based on probabilities).

## Implementation Details

### Batch Processing

The kernel efficiently handles batch processing:

```python
batch_size, vocab_size = probs.shape
```

All operations are vectorized to process entire batches simultaneously.

### Memory Efficiency

The implementation uses in-place operations where possible:
- `scatter_` for creating filtered distributions
- Efficient tensor operations to avoid unnecessary copies

### Parameter Validation

The implementation handles edge cases:
- Temperature of 1.0 bypasses scaling
- top_k of -1 bypasses top-k filtering
- top_p of 1.0 bypasses nucleus filtering

## Usage in Inference Engine

### Integration with Generation Loop

The sampling kernel integrates with the main inference loop:

```python
# Get logits from model
outputs = self.model(current_ids)
next_token_logits = outputs.logits[:, -1, :]  # Get last token logits

# Use sampling kernel for token selection
sampling_kernel = SamplingKernel()
next_token_id = sampling_kernel.sample(
    next_token_logits,
    temperature=request.temperature,
    top_p=request.top_p
)
```

### Parameter Control

Users can control generation characteristics through sampling parameters:

```python
# Creative generation (high randomness)
next_token = sampling_kernel.sample(logits, temperature=1.2, top_p=0.9)

# Focused generation (low randomness)
next_token = sampling_kernel.sample(logits, temperature=0.8, top_k=50)
```

## Sampling Strategy Comparison

### Temperature Sampling

**Use Case:** Controlling overall randomness
- **Low (0.5)**: More deterministic, focused output
- **Medium (1.0)**: Standard randomness
- **High (1.5)**: More diverse, creative output

### Top-K Sampling

**Use Case:** Preventing very low probability tokens
- **Small K (20-50)**: More focused, predictable output
- **Large K (200-500)**: More diverse, creative output
- **K = vocab_size**: Equivalent to no top-k filtering

### Top-P (Nucleus) Sampling

**Use Case:** Adaptive probability threshold
- **Low P (0.1-0.5)**: More focused, predictable output
- **Medium P (0.7-0.9)**: Good balance
- **High P (0.95-0.99)**: More diverse, creative output
- **P = 1.0**: Equivalent to no top-p filtering

## Performance Characteristics

### Computation Complexity

- **Time:** O(V log K) for Top-K, O(V log V) for Top-P, where V is vocabulary size
- **Space:** O(batch_size × vocab_size) for probability storage
- **GPU Optimized:** All operations use efficient PyTorch tensor operations

### Parallel Processing

The kernel efficiently handles parallel requests:
- Batched operations process multiple requests simultaneously
- Vectorized operations maximize GPU utilization
- Memory access patterns optimized for GPU architecture

## Advanced Features

### Combined Sampling Strategies

Multiple sampling strategies can be combined:

```python
# Apply temperature scaling, then top-k, then top-p
# (Order typically: temperature first, then top-k, then top-p)
```

### Dynamic Parameter Adjustment

Parameters can be adjusted per request:
- Different temperature for different request types
- Adaptive top-p based on context
- Request-specific sampling strategies

## Integration with SGLang Features

### Multi-Step Generation

The sampling kernel supports multi-step generation in SGLang-style systems:
- Consistent sampling strategy across generation steps
- Parameter maintenance for ongoing requests
- Batch processing with mixed request types

### Performance Monitoring

The kernel can be extended with:
- Sampling time measurement
- Distribution analysis
- Quality metrics for generated tokens

## Error Handling and Robustness

### Invalid Parameter Handling

The implementation handles:
- Negative temperatures (convert to positive)
- top_k larger than vocabulary size
- top_p outside [0,1] range
- Zero probability distributions

### Edge Cases

- Empty probability distributions
- Single-token vocabularies
- Very large batch sizes
- Mixed request parameter requirements

## Future Enhancements

### Advanced Sampling Strategies

Potential additions include:
- Min-P sampling (complement to Top-P)
- Tail-free sampling (TFS)
- Locally typical sampling
- Contrastive search

### Optimizations

Performance improvements could include:
- CUDA-optimized sampling kernels
- Quantized probability operations
- Caching for repeated requests
- Asynchronous sampling for better throughput

### Integration Features

Enhanced integration might include:
- Quality-based sampling adjustment
- Context-aware parameter selection
- Learned sampling strategies
- Multi-objective sampling

This sampling kernel provides the essential token selection functionality for LLM inference, enabling users to control generation quality and characteristics while maintaining high performance and efficiency.