# Radix Attention: SGLang-Style Prefix Sharing

## Overview

Radix Attention is a specialized attention mechanism designed to maximize computational efficiency by sharing computations for requests with common prefixes. This is a core component of SGLang-style inference engines, where multiple requests with similar starting text can reuse the same attention computations, dramatically reducing the total number of required operations.

## Core Concepts

### Prefix Sharing

Traditional inference processes each request independently, even if multiple requests start with identical text. Radix attention addresses this by:

- Identifying requests with common prefixes
- Sharing computation for the shared prefix tokens
- Only computing unique suffixes separately
- Reducing total computational overhead

### Radix Tree Structure

The attention mechanism uses a radix tree (trie) structure to organize token sequences:

```
Radix Tree Example:
                Root
               /    \
              A      B        <- First token: requests 1,2 have 'A', request 3 has 'B'
             / \    /   \
            C   D  E     F    <- Second token: requests 1,2 split, requests 3,4 split
           /   /   |     |
          G   H    I     J   <- Third token: continue branching

Computation Sharing:
- Tokens A and C are computed once for requests 1 and 2
- Tokens D and H are computed separately for request 2
```

## Implementation Architecture

### RadixAttentionBlock

The `RadixAttentionBlock` implements core attention functionality with radial awareness:

```python
class RadixAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, ...):
        # Initialize query/key/value projections
        self.q_proj = nn.Linear(hidden_size, total_hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, total_hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, total_hidden_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(total_hidden_dim, hidden_size, bias=False)
        
        # Rotary Position Embedding (RoPE) cache
        self.cos_cached = torch.ones((max_position_embeddings, head_dim))
        self.sin_cached = torch.ones((max_position_embeddings, head_dim))
```

### Key Components

#### 1. Linear Projections

The block implements standard attention projections:
- `q_proj`: Query projection matrix
- `k_proj`: Key projection matrix
- `v_proj`: Value projection matrix
- `o_proj`: Output projection matrix

These are used to transform input hidden states for attention computation.

#### 2. Rotary Position Embeddings (RoPE)

```python
def _setup_rope_embeddings(self):
    # Initialize RoPE for position-aware attention
    position_ids = torch.arange(self.max_position_embeddings, dtype=torch.float32)
    inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
    
    freqs = torch.einsum("i,j->ij", position_ids, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    
    self.cos_cached = emb.cos().to(dtype=torch.float16)
    self.sin_cached = emb.sin().to(dtype=torch.float16)
```

The RoPE embeddings ensure position information is preserved in shared computations.

#### 3. Forward Pass Implementation

The forward method handles radial attention computation:

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, ...]:
```

**Processing Steps:**

1. **QKV Projection**: Transform input states to query, key, value tensors
2. **Reshape**: Convert to multi-head format `[batch_size, num_heads, seq_len, head_dim]`
3. **RoPE Application**: Apply rotary embeddings considering position IDs
4. **KV-Cache Handling**: Manage past key-value states for incremental decoding
5. **Attention Computation**: Compute scaled dot-product attention
6. **Output Projection**: Transform results back to hidden size

## RadixAttentionWithPagedKVCache

The `RadixAttentionWithPagedKVCache` class integrates radial attention with paged memory management:

```python
class RadixAttentionWithPagedKVCache:
    def __init__(self, num_layers: int, num_heads: int, head_dim: int, ...):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks_per_request = max_blocks_per_request
        
        # Memory pools for key and value caches
        self.key_cache_pool = {}
        self.value_cache_pool = {}
        
        # Track blocks assigned to requests
        self.request_block_map = {}  # request_id -> [block_ids]
        self.next_block_index = 0
```

### Paged Cache Management

#### Block Allocation

```python
def allocate_blocks(self, request_id: str, num_blocks_needed: int) -> List[int]:
    # Allocate blocks and initialize cache tensors
    allocated_block_ids = []
    for _ in range(num_blocks_needed):
        block_id = self.next_block_index
        self.next_block_index += 1
        
        # Initialize key and value cache blocks
        self.key_cache_pool[block_id] = torch.zeros(
            self.block_size, self.num_heads, self.head_dim,
            dtype=torch.float16, device=device
        )
        self.value_cache_pool[block_id] = torch.zeros(
            self.block_size, self.num_heads, self.head_dim,
            dtype=torch.float16, device=device
        )
        allocated_block_ids.append(block_id)
    
    self.request_block_map[request_id] = allocated_block_ids
    return allocated_block_ids
```

#### Slot Appending

```python
def append_slot(self, key: torch.Tensor, value: torch.Tensor, request_id: str):
    # Append new key-value pairs to cache for a request
    if request_id not in self.request_block_map:
        # Allocate blocks for new request if needed
        self.allocate_blocks(request_id, 1)
    
    # Find appropriate block for appending
    # Handle block expansion when current block fills
    # Split across blocks if needed
```

The slot appending logic intelligently manages block usage and handles overflow conditions.

#### Cache Retrieval

```python
def get_kv_cache(
    self, request_ids: List[str], seq_lens: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Collect keys and values from blocks for specified requests
    # Handle multiple request retrieval efficiently
    # Format for attention computation
```

## SGLang-Style Optimizations

### 1. Shared Computation Efficiency

By sharing computation for common prefixes:

- Multiple requests compute the same prefix tokens only once
- Results are reused across all requests sharing the prefix
- Total computation complexity is reduced from sum of individual requests to union of token sequences

### 2. Memory Efficiency Integration

The radial attention integrates with paged KV-cache:

- Shared prefix tokens use the same cache blocks
- Reduces memory overhead for similar requests
- Enables more concurrent requests in memory-constrained environments

### 3. Prefix Tracking

The system maintains information about which tokens are shared:

- Block assignment tracking for each request
- Efficient retrieval of shared computation results
- Proper handling of request divergence after common prefixes

## Attention Computation Process

### Causal Attention Pattern

The radial attention maintains causality while enabling sharing:

```python
# Apply causal mask to prevent future token attention
attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)

if attention_mask is not None:
    expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    attn_weights = attn_weights.masked_fill(expanded_mask == 0, float("-inf"))

attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
```

### Rotary Position Embedding Application

RoPE ensures position information is preserved even in shared computations:

```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    # Apply rotary embeddings to queries and keys
    # Preserve relative position information in shared computation paths
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

## Performance Characteristics

### Computational Complexity

- **Traditional**: O(S×N×L²) where S=sequence length, N=number of requests, L=length of sequence
- **Radix Attention**: O(S×U²) where U is the union of all token sequences
- For similar requests, U << N×L, resulting in significant speedup

### Memory Complexity

- **Traditional**: O(N×L) for KV-cache
- **Radix**: O(U) for shared KV-cache
- Significant memory savings for requests with common prefixes

### Cache Efficiency

- Better cache utilization due to shared computation
- Reduced cache misses for similar token sequences
- More efficient GPU memory access patterns

## Integration with System Components

### Scheduler Integration

The attention mechanism works with the SGLang scheduler to:
- Identify requests with shared prefixes
- Group computation efficiently
- Update scheduler with completion status

### Memory Manager Integration

Integrates with the paged KV-cache system to:
- Allocate appropriate memory blocks
- Manage shared memory usage
- Track block usage for each request

### Model Integration

The attention mechanism fits into the overall model architecture to:
- Process input sequences efficiently
- Maintain compatibility with HuggingFace models
- Support both prefill and decode phases

## Usage in Inference Engine

### During Prefill Phase

For processing full prompts:
1. Identify common prefixes among new requests
2. Compute shared prefix tokens once
3. Compute unique suffixes separately
4. Store results in paged KV-cache

### During Decode Phase

For token generation:
1. Retrieve shared prefix states from cache
2. Process only the current token position
3. Update KV-cache with new states
4. Prepare for next token generation

## Challenges and Solutions

### 1. Prefix Divergence

When requests with common prefixes diverge:

- Properly handle cache invalidation
- Maintain separate computation paths
- Update block mappings accordingly

### 2. Memory Management Complexity

Managing shared memory across multiple requests:

- Sophisticated block allocation algorithms
- Reference counting for shared blocks
- Proper cleanup when requests complete

### 3. Synchronization Requirements

Ensuring consistency across shared computations:

- Proper isolation between request states
- Consistent block state management
- Race condition prevention in multi-threaded contexts

## Future Enhancements

### Advanced Sharing Strategies

- Semantic similarity-based sharing beyond exact prefix matching
- Dynamic grouping based on runtime computation patterns
- Machine learning-based prediction of sharing benefits

### Optimization Techniques

- Hardware-specific optimizations for different GPU architectures
- Quantization techniques for more efficient computation
- Asynchronous computation scheduling for better GPU utilization

### Integration Improvements

- More sophisticated radix tree algorithms
- Advanced caching strategies
- Hybrid attention mechanisms combining radial with other approaches