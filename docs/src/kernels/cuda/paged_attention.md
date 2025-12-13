# Paged Attention (`kernels/cuda/paged_attention.cu`)

## Concept

Compute attention where K and V are stored in non-contiguous blocks.

## Implementation Goal

Implement `paged_attention_kernel`:

### Inputs

- `block_tables`: A tensor mapping `[request_id, logical_block_idx] -> physical_block_idx`.

### Logic

1.  **Thread Mapping**: Each thread block handles one sequence (request).
2.  **Gathering**:
    - Instead of `K[i]`, we must compute the physical address.
    - `block_number = block_tables[request_id][token_index / block_size]`
    - `block_offset = token_index % block_size`
    - `physical_addr = base_ptr + block_number * stride + block_offset`
3.  **Attention**:
    - Load K, V using the calculated physical addresses.
    - Compute Attention as usual.
