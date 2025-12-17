# Radix Attention CUDA Kernel (`radix_attention.cu`)

The `radix_attention.cu` module provides an optimized CUDA kernel implementation for Radix Attention. This kernel focuses on leveraging radix tree indices for efficient prefix sharing during attention computation, aiming to reduce redundant calculations in scenarios like continuous batching in Large Language Models (LLMs).

## Key Concepts

*   **Radix Attention**: An attention mechanism designed to work efficiently with prefix sharing. It utilizes a Radix Tree (implemented in `radix_tree.py`) to identify common prefixes among sequences, allowing shared computations for those prefixes.
*   **Prefix Sharing**: A technique where common initial token sequences (prefixes) across different requests are processed only once, and their Key-Value (KV) states are reused. This significantly improves throughput, especially with many similar prompts.
*   **Radix Tree Indices (`radix_indices_ptr`)**: These indices are presumably used to guide the kernel on which parts of the KV cache correspond to shared prefixes, allowing it to efficiently access and aggregate relevant KV states.

## Macros

*   `WARP_SIZE`: Standard CUDA warp size (typically `32`), used for thread organization.
*   `MAX_SHARED_MEMORY`: Defines the maximum amount of shared memory available per CUDA block (e.g., `48KB`).

## `radix_attention_kernel_impl` (CUDA Kernel)

This is the core CUDA kernel for computing Radix Attention. It processes queries and accesses keys and values from a shared KV cache, guided by radix indices and block tables.

*   **Purpose**: To compute the attention output for individual tokens, efficiently utilizing a paged KV cache with considerations for prefix sharing.
*   **Function Parameters**:
    *   `query_ptr` (const `float*`): Pointer to the query tensor data (`[num_tokens, num_heads, head_dim]`). Each `num_tokens` here might represent a single token being processed in an autoregressive step.
    *   `key_cache_ptr` (const `float*`): Pointer to the physical key cache memory (`[num_blocks, block_size, num_heads, head_dim]`).
    *   `value_cache_ptr` (const `float*`): Pointer to the physical value cache memory (`[num_blocks, block_size, num_heads, head_dim]`).
    *   `block_tables_ptr` (const `int*`): Pointer to the block table mapping logical sequence positions to physical block IDs (`[num_tokens, max_blocks_per_seq]`).
    *   `context_lens_ptr` (const `int*`): Pointer to the actual context length for each token being processed (`[num_tokens]`).
    *   `radix_indices_ptr` (const `int*`): Pointer to indices that inform the kernel about prefix sharing opportunities. (The exact interpretation would depend on the `radix_tree.py`'s output).
    *   `output_ptr` (`float*`): Pointer to the output tensor data (`[num_tokens, num_heads, head_dim]`).
    *   `scale` (const `float`): Scaling factor for attention scores (e.g., `1 / sqrt(head_dim)`).
    *   `num_tokens` (const `int`): Total number of tokens being processed in the current kernel launch.
    *   `num_heads` (const `int`): Number of attention heads.
    *   `head_dim` (const `int`): Dimension of each attention head.
    *   `block_size` (const `int`): Size of each KV-cache block.
    *   `max_blocks_per_seq` (const `int`): Maximum number of blocks a single sequence can occupy.
*   **Internal Logic**:
    1.  **Thread Indexing**: Threads are mapped to process a specific `token_idx`, `head_idx`, and `d_idx` (dimension within the head).
    2.  **Context and Radix Information**: Retrieves the `context_len` for the current token and its `radix_idx`, which guides prefix sharing.
    3.  **Local Accumulators**: Initializes `sum_exp`, `max_logit`, and `out_val` for numerically stable softmax calculation.
    4.  **Query Value**: Extracts the query value for the current thread's dimension.
    5.  **Context Iteration**: Loops through all relevant past tokens up to `context_len`.
    6.  **KV Cache Access**: Uses `block_tables_ptr` to find the physical block ID and position within the block for each past token, then retrieves the corresponding key and value from `key_cache_ptr` and `value_cache_ptr`.
    7.  **Attention Score**: Computes the dot product (`query_val * key_val * scale`).
    8.  **Numerical Stability (Softmax)**: Implements the "max-logit trick" to prevent numerical overflow/underflow during softmax calculation by iteratively updating `max_logit`, `sum_exp`, and `out_val`.
    9.  **Result Storage**: The final attention output for the current token, head, and dimension is written to `output_ptr`.

## `radix_attention_kernel` (CUDA C++ Wrapper)

This C++ function serves as a wrapper to prepare input tensors, configure kernel launch parameters, and invoke the `radix_attention_kernel_impl` CUDA kernel.

*   **Purpose**: Provides a PyTorch-compatible interface to execute the Radix Attention CUDA kernel.
*   **Parameters**:
    *   `output` (torch::Tensor): The output tensor for attention results.
    *   `query` (torch::Tensor): Input query tensor.
    *   `key_cache` (torch::Tensor): The global key cache tensor.
    *   `value_cache` (torch::Tensor): The global value cache tensor.
    *   `radix_indices` (torch::Tensor): Tensor containing radix tree indices.
    *   `block_size` (int): The size of each KV-cache block.
*   **Internal Logic**:
    1.  **Dimension Extraction**: Retrieves `num_tokens`, `num_heads`, `head_dim` from input tensors. `max_blocks_per_seq` is currently derived from `key_cache.size(0)`, which might need adjustment based on the actual design of the KV cache and block tables.
    2.  **Grid and Block Configuration**: Sets up the `grid` (number of blocks) and `block` (threads per block) dimensions for the kernel launch. The `block.x` dimension is limited to `1024` threads.
    3.  **Simplified `block_tables` and `context_lens`**: For this example, `block_tables` and `context_lens` are simplified and created as dummy tensors. In a full implementation, these would be populated by a scheduler or KV cache manager.
    4.  **Scaling Factor**: Computes the `scale` factor (`1.0f / sqrtf(static_cast<float>(head_dim))`).
    5.  **Kernel Launch**: Invokes the `radix_attention_kernel_impl` with the configured grid, block, shared memory size (0, as shared memory is dynamically declared within the kernel), and CUDA stream.
    6.  **Error Checking**: Includes `cudaGetLastError()` to catch and report any CUDA kernel launch errors.
