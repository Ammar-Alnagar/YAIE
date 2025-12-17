# Memory Operations CUDA Kernel (`memory_ops.cu`)

The `memory_ops.cu` module provides highly optimized CUDA kernels for fundamental memory operations, specifically designed to efficiently manage the Key-Value (KV) cache in large language models (LLMs). These operations are critical for implementing advanced KV-cache management strategies such as prefix sharing, request re-scheduling, and dynamic memory allocation.

## Key Concepts

*   **Efficient KV-Cache Management**: Operations like copying and swapping KV-cache blocks are essential for optimizing LLM inference. For example, when multiple requests share a common prefix, the KV states for that prefix can be copied rather than recomputed. Similarly, swapping blocks can be part of a sophisticated memory defragmentation or re-prioritization scheme.
*   **CUDA Kernels**: Direct GPU implementations leverage parallel processing for significant speedup compared to CPU-based memory operations, especially for large tensors.

## Macros

*   `BLOCK_SIZE`: A generic block size, likely used to define the dimensions of KV-cache blocks.
*   `WARP_SIZE`: The standard CUDA warp size (`32`), fundamental for organizing threads and optimizing memory access patterns.
*   `MAX_THREADS_PER_BLOCK`: Defines the maximum number of threads that can be launched within a single CUDA block (e.g., `1024`).

## `copy_blocks_kernel_impl` (CUDA Kernel)

This kernel is designed to copy the contents of multiple source KV-cache blocks to multiple destination KV-cache blocks. This is particularly useful for duplicating prefixes for new requests or for state migration.

*   **Purpose**: To perform parallel copying of data from one set of KV-cache blocks to another.
*   **Function Parameters**:
    *   `src_key_ptr` (const `float*`): Pointer to the global memory location of source key data.
    *   `src_value_ptr` (const `float*`): Pointer to the global memory location of source value data.
    *   `dst_key_ptr` (`float*`): Pointer to the global memory location of destination key data.
    *   `dst_value_ptr` (`float*`): Pointer to the global memory location of destination value data.
    *   `block_mapping_ptr` (const `int*`): A linear array containing pairs of `[src_block_id, dst_block_id]` for each copy operation (`[num_mappings * 2]`).
    *   `num_blocks` (const `int`): The number of block copy operations to perform (i.e., `num_mappings`).
    *   `block_size` (const `int`): The number of tokens in each KV-cache block.
    *   `num_heads` (const `int`): The number of attention heads.
    *   `head_dim` (const `int`): The dimension of each attention head.
*   **Internal Logic**:
    1.  **Thread Indexing**: `mapping_idx` identifies which block-copy operation a block of threads is responsible for, and `thread_idx` identifies individual threads within that block.
    2.  **Block ID Retrieval**: The kernel uses `block_mapping_ptr` to get the `src_block_id` and `dst_block_id` for the current copy operation.
    3.  **Cooperative Copy**: Each thread within a block cooperatively copies a portion of the `elements_per_block` from the source `key_cache_ptr` and `value_cache_ptr` to the respective destination pointers.

## `copy_single_block_kernel_impl` (CUDA Kernel)

This kernel is similar to `copy_blocks_kernel_impl` but is specialized for copying the contents of a single source KV-cache block to a single destination KV-cache block. This might be used in scenarios where only one block needs to be duplicated or moved.

*   **Purpose**: To efficiently copy all elements from one KV-cache block to another.
*   **Function Parameters**:
    *   `src_key_ptr` (const `float*`): Pointer to the global memory location of source key data.
    *   `src_value_ptr` (const `float*`): Pointer to the global memory location of source value data.
    *   `dst_key_ptr` (`float*`): Pointer to the global memory location of destination key data.
    *   `dst_value_ptr` (`float*`): Pointer to the global memory location of destination value data.
    *   `src_block_id` (const `int`): The ID of the source block.
    *   `dst_block_id` (const `int`): The ID of the destination block.
    *   `block_size` (const `int`): The number of tokens in each KV-cache block.
    *   `num_heads` (const `int`): The number of attention heads.
    *   `head_dim` (const `int`): The dimension of each attention head.
*   **Internal Logic**:
    1.  **Thread Indexing**: `tid` directly maps to an element index within the block.
    2.  **Direct Copy**: Each thread calculates its specific `src_offset` and `dst_offset` and copies the corresponding key and value elements.

## `swap_blocks_kernel_impl` (CUDA Kernel)

This kernel facilitates the swapping of content between two KV-cache blocks. This can be used for defragmentation, re-ordering of blocks, or in complex scheduling algorithms.

*   **Purpose**: To atomically swap the contents of specified KV-cache blocks.
*   **Function Parameters**:
    *   `key_cache_ptr` (`float*`): Pointer to the global memory location of key data (both source and destination).
    *   `value_cache_ptr` (`float*`): Pointer to the global memory location of value data (both source and destination).
    *   `block_mapping_ptr` (const `int*`): A linear array containing pairs of `[block_id_1, block_id_2]` for each swap operation (`[num_swaps * 2]`).
    *   `num_swaps` (const `int`): The number of block swap operations to perform.
    *   `block_size` (const `int`): The number of tokens in each KV-cache block.
    *   `num_heads` (const `int`): The number of attention heads.
    *   `head_dim` (const `int`): The dimension of each attention head.
*   **Internal Logic**:
    1.  **Thread Indexing**: `swap_idx` identifies which block-swap operation a block of threads is responsible for.
    2.  **Block ID Retrieval**: `block_mapping_ptr` provides the two `block_id`s whose contents are to be swapped.
    3.  **Cooperative Swap**: Threads cooperatively swap elements between the two identified blocks for both keys and values, using temporary variables to ensure correct exchange.

## `copy_blocks_kernel` (CUDA C++ Wrapper)

This C++ function is a wrapper to launch the `copy_blocks_kernel_impl` CUDA kernel from PyTorch.

*   **Purpose**: Provides a user-friendly interface in Python (via PyTorch) to execute the `copy_blocks_kernel_impl`.
*   **Parameters**:
    *   `key_cache` (torch::Tensor): The PyTorch tensor representing the key cache.
    *   `value_cache` (torch::Tensor): The PyTorch tensor representing the value cache.
    *   `block_mapping` (torch::Tensor): A PyTorch tensor containing the `[src_block_id, dst_block_id]` pairs.
    *   `num_mappings` (int): The number of copy operations.
*   **Internal Logic**: Extracts dimensions from `key_cache`, configures the CUDA grid and block dimensions, and launches `copy_blocks_kernel_impl` on the current CUDA stream. Includes error checking.

## `swap_blocks_kernel` (CUDA C++ Wrapper)

This C++ function is a wrapper to launch the `swap_blocks_kernel_impl` CUDA kernel from PyTorch.

*   **Purpose**: Provides a user-friendly interface in Python (via PyTorch) to execute the `swap_blocks_kernel_impl`.
*   **Parameters**:
    *   `key_cache` (torch::Tensor): The PyTorch tensor representing the key cache.
    *   `value_cache` (torch::Tensor): The PyTorch tensor representing the value cache.
    *   `block_mapping` (torch::Tensor): A PyTorch tensor containing the `[block_id_1, block_id_2]` pairs for swapping.
    *   `num_swaps` (int): The number of swap operations.
*   **Internal Logic**: Extracts dimensions from `key_cache`, configures the CUDA grid and block dimensions, and launches `swap_blocks_kernel_impl` on the current CUDA stream. Includes error checking.
