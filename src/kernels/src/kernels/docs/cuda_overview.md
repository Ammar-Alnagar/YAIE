# CUDA Kernels Overview (`cuda/` directory)

The `cuda/` directory within the `src/kernels` module houses highly optimized implementations of performance-critical operations using NVIDIA's CUDA platform. These CUDA kernels (`.cu` files) are designed to leverage the parallel processing capabilities of GPUs, providing significant speedups for demanding tasks in large language model (LLM) inference.

For development, testing, and environments without CUDA-enabled GPUs, corresponding CPU fallback implementations (`_cpu.cpp` files) are often provided. These CPU versions offer functional equivalence, albeit with expectedly lower performance.

## Structure and Purpose:

Each pair of `.cu` and `_cpu.cpp` files typically corresponds to a specific optimized operation:

*   **`flash_attention.cu` / `flash_attention_cpu.cpp`**:
    Implementations of Flash Attention, a highly efficient attention mechanism that reduces memory I/O and improves speed by fusing several attention operations into a single CUDA kernel. The `.cu` file is the GPU-optimized version, and the `_cpu.cpp` is its CPU counterpart.

*   **`memory_ops.cu` / `memory_ops_cpu.cpp`**:
    Contains optimized memory manipulation routines. These kernels are essential for efficient data movement and transformation on the GPU, crucial for managing large tensors in LLMs. The `.cu` file is the GPU-optimized version, and the `_cpu.cpp` is its CPU counterpart.

*   **`paged_attention.cu` / `paged_attention_cpu.cpp`**:
    Implementations of Paged Attention, an advanced KV-cache management technique that improves GPU memory utilization by storing KV-cache entries in fixed-size "pages" or blocks. This allows for non-contiguous memory allocation and more efficient handling of variable sequence lengths. The `.cu` file is the GPU-optimized version, and the `_cpu.cpp` is its CPU counterpart.

*   **`radix_attention.cu` / `radix_attention_cpu.cpp`**:
    Implementations specifically for the Radix Attention mechanism, likely focusing on its most computationally intensive parts. This could involve optimized kernel launches for the specific data access patterns of Radix Attention. The `.cu` file is the GPU-optimized version, and the `_cpu.cpp` is its CPU counterpart.

*   **`README.md`**:
    This file likely contains specific instructions, build procedures, or additional details pertaining to the CUDA implementations within this directory. It should be consulted for more in-depth information about compiling and integrating these kernels.

By providing both CUDA-accelerated and CPU fallback implementations, the `kernels` module ensures broad compatibility while maximizing performance on capable hardware.