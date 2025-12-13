# Memory Manager (`kernels/kv_cache.py`)

The `KVCacheManager` is the "Allocator". It manages the Paged KV Cache.

## The Block Table

Just like an OS manages RAM pages, this manager tracks GPU memory blocks.

- **Physical Blocks**: Fixed-size tensors in GPU memory (e.g., `[num_blocks, block_size, head_dim]`).
- **Logical Slots**: The token positions in a sequence.

## Key Methods

- `allocate_blocks(request)`: Finds free physical blocks and assigns them to a request.
- `free_blocks(request)`: Returns blocks to the free pool.
- `get_block_table(request)`: Returns the mapping `[logical_idx -> physical_idx]` for the attention kernel.

**Student Task**: You will implement the allocation logic in the Python Kernels phase.
