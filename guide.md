# Mini-YAIE Kernels Implementation Guide

This document outlines the kernels that need to be implemented in the Mini-YAIE inference engine, following SGLang architecture and concepts. Each section describes what the kernel does, why it's important, and implementation guidance.

## 1. Radix Attention Kernels (SGLang Core)

### 1.1 RadixAttentionBlock

**Purpose**: SGLang's radial attention mechanism with prefix sharing for efficient batch processing of requests with common prefixes.

**How it Works**:
- Represents request prefixes as a tree structure (radix tree)
- Shares computation for requests with common prefixes
- Reduces redundant computation in continuous batching scenarios

**Visual Representation**:

```
Radix Tree Structure:
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
- This reduces total computation compared to separate processing
```

**Implementation Steps**:
1. Parse input requests to identify common prefixes
2. Build radix tree structure from shared prefixes
3. Implement forward pass that computes shared nodes once
4. Implement backward pass for gradient computation
5. Handle KV-cache management for shared computation

**Code Structure**:
```python
class RadixAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, max_position_embeddings):
        # Initialize query/key/value projections
        # Initialize rotary embedding components
        # Initialize radix tree management utilities
    
    def forward(self, hidden_states, radix_tree_info, attention_mask, position_ids):
        # Step 1: Apply QKV projections
        # Step 2: Apply RoPE embeddings
        # Step 3: Use radix tree to identify shared computation
        # Step 4: Compute attention for shared vs. unique parts
        # Step 5: Manage KV-cache for next iteration
```

**Why it's Important**:
- Dramatically reduces redundant computation in continuous batching
- Enables efficient processing of multiple requests with similar prefixes
- Core to SGLang's performance advantages

### 1.2 RadixAttentionWithPagedKVCache

**Purpose**: Radix attention combined with paged KV-cache management for memory efficiency.

**How it Works**:
- Combines radix attention with efficient KV-cache paging
- Manages shared KV-cache blocks for requests with common prefixes
- Reduces memory usage while maintaining computation sharing benefits

**Visual Representation**:

```
Paged KV Cache with Radix Tree:
Radix Tree:
    Root -> Token A -> Token C -> Token G
                     -> Token D -> Token H

Paged Memory:
Physical Blocks: [Blk1][Blk2][Blk3][Blk4][Blk5][...]
                 [KV_A][KV_C][KV_D][KV_G][KV_H]

Block Table for Shared Computation:
- Root->A: Blk1 (shared by multiple requests)
- A->C: Blk2 (shared by requests with prefix "AC")
- A->D: Blk3 (used by requests with prefix "AD")  
- C->G: Blk4 (specific to request with prefix "ACG")
- D->H: Blk5 (specific to request with prefix "ADH")
```

**Implementation Steps**:
1. Implement paged memory allocator for KV-cache
2. Create block reference counting for shared prefixes
3. Handle block allocation/deallocation based on radix tree
4. Implement efficient block lookup during attention computation
5. Add memory management policies for cache eviction

## 2. SGLang-Style Prefill & Decode Optimization

### 2.1 Chunked Prefill

**Purpose**: Efficiently process long initial prompts using chunked processing to reduce memory usage.

**How it Works**:
- Splits long prompts into smaller chunks
- Processes chunks efficiently without materializing full attention
- Reduces peak memory usage during prefill phase

**Visual Representation**:

```
Long Prompt (1000 tokens):
Input: [T1, T2, T3, ..., T1000]

Traditional Prefill:
- All 1000x1000 attention matrix: O(1000²) memory
- Peak memory usage: Very high

Chunked Prefill:
- Split into 10 chunks of 100 tokens each
- Process: [T1-T100], [T101-T200], ..., [T901-T1000]
- Each chunk: 100x100 attention: O(100²) memory
- Cache intermediate KV states between chunks
- Combine results efficiently
```

**Implementation Steps**:
1. Implement chunking logic for long prompts
2. Maintain KV-cache across chunks
3. Handle cross-chunk attention efficiently
4. Optimize memory allocation for chunked processing

### 2.2 Decode Phase Optimization

**Purpose**: Optimize single-token generation phase for maximum throughput.

**How it Works**:
- Process one token at a time (current position only)
- Efficiently retrieve from paged KV-cache
- Minimize memory access for best performance

**Implementation Steps**:
1. Implement single-token attention computation
2. Optimize KV-cache retrieval from paged memory
3. Handle batched decode requests efficiently
4. Optimize for GPU memory access patterns

## 3. Memory Management Kernels

### 3.1 Page-Based KV Cache Management

**Purpose**: Efficient memory management for KV-cache using page-based allocation.

**How it Works**:
- KV-cache stored in fixed-size pages
- Each request references required pages through block table
- Reduces memory fragmentation compared to contiguous allocation

**Visual Representation**:

```
Memory Layout - Page-based:
Global Page Pool (GPU Memory):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│Page0│Page1│Page2│Page3│Page4│Page5│Page6│ ... │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

Request 1 (tokens 0-150): [Page0, Page1] (0-63, 64-127 in Page0, 128-150 in Page1)  
Request 2 (tokens 0-80):  [Page2, Page4] (non-contiguous but efficient)
Request 3 (tokens 0-200): [Page3, Page5, Page6, ...]

Block Table per Request:
Request 1: [0, 1]  (maps to Page0, Page1)
Request 2: [2, 4]  (maps to Page2, Page4) 
Request 3: [3, 5, 6, ...] (maps to Page3, Page5, Page6, ...)
```

**Implementation Steps**:
1. Implement fixed-size page allocator
2. Create block table management system
3. Handle page allocation and deallocation efficiently
4. Implement page swapping for memory-constrained scenarios

### 3.2 Radix Tree Management

**Purpose**: Efficient management of radix trees for tracking request prefixes.

**How it Works**:
- Maintains tree structure of shared prefixes
- Enables efficient lookup during request processing
- Manages tree updates when requests progress

**Implementation Steps**:
1. Implement radix tree node structure
2. Create efficient insertion/deletion methods
3. Handle tree balancing and optimization
4. Implement cache for frequently accessed paths

## 4. Position Embedding Kernels

### 4.1 RoPE with Radix Attention

**Purpose**: Apply rotary embeddings efficiently in the context of radix attention.

**How it Works**:
- Compute RoPE for each token position
- Apply embeddings considering shared computation paths
- Optimize for both shared and unique prefix computations

**Implementation Steps**:
1. Precompute RoPE embeddings for all possible positions
2. Apply embeddings based on actual token positions in radix tree
3. Handle position ID management for shared prefixes
4. Optimize for GPU computation patterns

## 5. SGLang-Specific Optimizations

### 5.1 Multi-Step Attention

**Purpose**: Process multiple decoding steps in a single kernel call for efficiency.

**How it Works**:
- Instead of single-step decoding, process multiple steps
- Reduces kernel launch overhead
- Improves GPU utilization

**Implementation Steps**:
1. Implement multi-step processing logic
2. Manage KV-cache updates across multiple steps
3. Handle stopping conditions within multi-step execution
4. Balance performance gains with memory requirements

### 5.2 Request Scheduling with Radix

**Purpose**: Optimize request scheduling based on radix tree structure.

**How it Works**:
- Schedule requests with common prefixes together
- Maximize computation sharing opportunities
- Balance between sharing benefits and scheduling efficiency

**Implementation Steps**:
1. Analyze radix tree structure for scheduling opportunities
2. Implement efficient batch formation logic
3. Handle request preemption and rescheduling
4. Optimize for throughput and latency trade-offs

## Implementation Order Recommendation (SGLang Focus)

1. **Start with core radix components**:
   - Radix tree data structure
   - Basic KV-cache management
   - Simple attention (without radix sharing initially)

2. **Implement radix attention fundamentals**:
   - RadixAttentionBlock with basic sharing
   - Paged KV-cache integration
   - Position embedding with radix support

3. **Add optimization layers**:
   - Chunked prefill for long prompts
   - Multi-step decoding
   - Advanced memory management

4. **Fine-tune for SGLang patterns**:
   - Request scheduling with prefix awareness
   - Performance optimization for common use cases
   - Integration with the engine scheduler

## Testing Strategy for SGLang Kernels

1. **Test radix sharing correctness**: Verify that shared computation produces identical results to separate computation
2. **Benchmark memory efficiency**: Measure memory savings from prefix sharing and paged cache
3. **Validate performance gains**: Compare throughput with naive batching approaches
4. **Integration testing**: Ensure kernels work with the full engine pipeline

## Resources for SGLang Implementation

- SGLang paper: https://arxiv.org/abs/2308.07561
- SGLang GitHub: https://github.com/sgl-project/sglang
- Efficient memory management techniques
- Radix tree data structure implementations