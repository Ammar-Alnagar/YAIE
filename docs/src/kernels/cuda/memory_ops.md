# Memory Operations (`kernels/cuda/memory_ops.cu`)

## Concept

Moving data between different GPU memory locations is a frequent operation in Paged Attention.

## Implementation Goal

Implement `copy_blocks_kernel`:

### Signature

```cpp
void copy_blocks_kernel(
    torch::Tensor key_cache,      // [num_blocks, block_size, head_dim]
    torch::Tensor value_cache,    // [num_blocks, block_size, head_dim]
    torch::Tensor block_mapping,  // [num_mappings, 2] (src, dst)
    int num_mappings
);
```

### Logic

1.  **Parallelism**: Launch one thread per token to copy.
2.  **Indexing**:
    - `mapping_idx = blockIdx.x`
    - `src_block = block_mapping[mapping_idx][0]`
    - `dst_block = block_mapping[mapping_idx][1]`
3.  **Copy**:
    - Read `key/value` from `src_block` at `threadIdx` offset.
    - Write to `dst_block`.
