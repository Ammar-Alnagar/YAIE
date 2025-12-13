# KV Cache Manager (`kernels/kv_cache.py`)

## Concept

Manage a pool of fixed-size integer IDs representing GPU memory blocks.

## Implementation Goal

Implement `KVCacheManager`:

### 1. `__init__`

- Create a list of all available block indices: `free_blocks = [0, 1, ..., N-1]`.
- Initialize an empty mapping `request_to_blocks = {}`.

### 2. `allocate(request, num_tokens)`

- Calculate blocks needed: `ceil(num_tokens / block_size)`.
- Pop that many indices from `free_blocks`.
- Store them in `request_to_blocks`.
- **Edge Case**: If not enough blocks, raise OutOfMemory (or trigger eviction).

### 3. `free(request)`

- Retrieve the blocks owned by the request.
- Append them back to `free_blocks`.
- Delete the request mapping.
