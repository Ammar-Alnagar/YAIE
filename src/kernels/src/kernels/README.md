# Kernels Module

The `kernels` module in Mini-YAIE serves as the core computational engine, encapsulating highly optimized operations critical for efficient execution of large language models (LLMs). It provides specialized implementations for attention mechanisms, KV cache management, and token sampling, leveraging both CPU and CUDA for performance.

## Main Components:

*   **API (`api.py`)**: Defines the public interface for interacting with the kernel functionalities.
*   **Engine (`engine.py`)**: Orchestrates the execution of various kernels, managing the overall computational flow.
*   **KV Cache (`kv_cache.py`)**: Manages the Key-Value cache, essential for optimizing attention computations in sequence generation.
*   **Radix Attention (`radix_attention.py`, `radix_tree.py`)**: Implements the Radix Attention mechanism, potentially with a `radix_tree` for efficient lookups or specific data structures.
*   **Sampling (`sampling.py`)**: Provides diverse strategies for sampling the next token during the language model's generation process.
*   **CUDA Implementations (`cuda/`)**: Contains highly optimized CUDA kernels for performance-critical operations like Flash Attention, Paged Attention, Radix Attention, and memory operations, with CPU fallbacks.
*   **Integration Tests (`test_kernels_integration.py`)**: Ensures the correct functionality and integration of the kernel components.

This module is designed for high performance and serves as the backbone for the Mini-YAIE LLM inference.