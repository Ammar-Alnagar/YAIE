# CUDA Kernels Directory

This directory contains CUDA kernel implementations that need to be developed as part of the educational exercise.

## Planned Kernels:

1. **flash_attention.cu** - FlashAttention implementation with forward and backward passes
2. **paged_attention.cu** - Paged attention for efficient KV-cache management  
3. **radix_attention.cu** - Radix attention with prefix sharing
4. **memory_ops.cu** - Memory management and utility kernels
5. **rope_embeddings.cu** - Rotary position embedding kernels
6. **rms_norm.cu** - RMS normalization kernels
7. **activation_kernels.cu** - SiLU and other activation function kernels

Each `.cu` file also needs a corresponding `.cpp` file for the Python bindings using PyTorch's extension system.

These are placeholder files that you will implement as part of the learning exercise.