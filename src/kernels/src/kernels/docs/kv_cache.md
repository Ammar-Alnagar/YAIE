# KV-Cache Module (`kv_cache.py`)

The `kv_cache.py` module implements a paged Key-Value (KV) cache system, which is crucial for efficient memory utilization in large language models (LLMs), especially during long sequence generation and continuous batching. This system avoids redundant computation of KV states for tokens that have already been processed.

## `KVCacheBlock` Class

Represents a single, fixed-size block of memory within the paged KV cache. Each block stores a portion of the key and value tensors for a given sequence.

### `__init__(self, block_id: int, size: int, num_heads: int, head_dim: int, dtype=torch.float16)`

Initializes a `KVCacheBlock` instance.

*   **Parameters**:
    *   `block_id` (int): A unique identifier for this block.
    *   `size` (int): The maximum number of tokens (or token positions) that this block can hold.
    *   `num_heads` (int): The number of attention heads in the model.
    *   `head_dim` (int): The dimensionality of each attention head.
    *   `dtype` (torch.dtype, default: `torch.float16`): The data type for the key and value tensors.
*   **Attributes Initialized**:
    *   `self.block_id`, `self.size`, `self.num_heads`, `self.head_dim`, `self.dtype`.
    *   `self.keys` (torch.Tensor or None): Placeholder for the key tensor data, initialized to `None`.
    *   `self.values` (torch.Tensor or None): Placeholder for the value tensor data, initialized to `None`.

### `allocate(self)`

Allocates GPU memory for the `keys` and `values` tensors within the block. This method is called when a block is assigned to store actual KV data. If a CUDA-enabled GPU is available, memory is allocated on the GPU; otherwise, it defaults to CPU.

## `KVCacheManager` Class

Manages the collection of `KVCacheBlock` instances, orchestrating their allocation, deallocation, and data transfer.

### `__init__(self, num_blocks: int, block_size: int, num_heads: int, head_dim: int, dtype=torch.float16)`

Initializes the KV cache manager.

*   **Parameters**:
    *   `num_blocks` (int): The total number of `KVCacheBlock` instances to pre-allocate.
    *   `block_size` (int): The size (in tokens) of each individual `KVCacheBlock`.
    *   `num_heads` (int): The number of attention heads across all blocks.
    *   `head_dim` (int): The dimensionality of each attention head.
    *   `dtype` (torch.dtype, default: `torch.float16`): The data type for KV tensors in all blocks.
*   **Attributes Initialized**:
    *   `self.blocks` (List[KVCacheBlock]): A list containing all pre-allocated `KVCacheBlock` instances.
    *   `self.free_block_list` (List[int]): A list of `block_id`s that are currently available for allocation.
    *   `self.block_tables` (dict): A mapping from `request_id` (str) to a list of `block_id`s, tracking which blocks belong to which active request.

### `allocate_blocks(self, request_id: str, num_tokens: int) -> List[int]`

Allocates a sufficient number of `KVCacheBlock`s from the `free_block_list` to accommodate a new request or extend an existing one for `num_tokens`.

*   **Parameters**:
    *   `request_id` (str): The unique identifier for the request.
    *   `num_tokens` (int): The number of tokens for which memory is needed.
*   **Returns**:
    *   `List[int]`: A list of `block_id`s that have been allocated to the `request_id`.
*   **Raises**:
    *   `RuntimeError`: If there are not enough free blocks available.

### `free_blocks(self, request_id: str)`

Deallocates all `KVCacheBlock`s associated with a given `request_id`, returning them to the `free_block_list`. This method also optionally clears the tensor data within the freed blocks to reclaim GPU memory.

*   **Parameters**:
    *   `request_id` (str): The unique identifier of the request whose blocks are to be freed.

### `copy_blocks(self, src_block_ids: List[int], dst_block_ids: List[int])`

Copies key-value data from a list of source blocks to a list of destination blocks. This operation is fundamental for features like prefix sharing, where common prefixes of different requests can share the same KV states, or for re-scheduling requests.

*   **Parameters**:
    *   `src_block_ids` (List[int]): A list of `block_id`s from which KV data will be copied.
    *   `dst_block_ids` (List[int]): A list of `block_id`s to which KV data will be copied.
*   **Raises**:
    *   `ValueError`: If the length of `src_block_ids` and `dst_block_ids` do not match.

### `get_kv_tensors(self, block_ids: List[int], seq_lens: List[int]) -> Tuple[torch.Tensor, torch.Tensor]`

Retrieves concatenated key and value tensors from a given list of `block_id`s. This method assembles the KV states for a batch of sequences, respecting their individual lengths.

*   **Parameters**:
    *   `block_ids` (List[int]): A list of `block_id`s that contain the desired KV data.
    *   `seq_lens` (List[int]): A list of sequence lengths, where each entry corresponds to the actual length of the sequence that the `block_ids` belong to.
*   **Returns**:
    *   `Tuple[torch.Tensor, torch.Tensor]`: A tuple containing two concatenated tensors: `(keys, values)`. These tensors are shaped for direct use in attention mechanisms.
