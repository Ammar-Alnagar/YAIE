# CUDA Kernels for Mini-YAIE

This directory contains the CUDA kernel implementations for Mini-YAIE, designed to provide SGLang-style performance optimizations for LLM inference.

## Kernel Overview

### Memory Operations (`memory_ops.cu`)
- Block copying for paged KV-cache management
- Memory allocation and deallocation utilities
- GPU-accelerated memory operations

### Paged Attention (`paged_attention.cu`)
- Attention computation with paged key-value cache
- GPU-optimized access patterns for paged memory
- Support for shared prefix computation

### Flash Attention (`flash_attention.cu`)
- Memory-efficient attention using tiling
- Online softmax computation for numerical stability
- Optimized for both prefill and decode phases

### Radix Operations (`radix_ops.cu`)
- GPU-accelerated prefix matching
- Radix tree traversal on GPU
- Shared computation identification

## Build Requirements

- CUDA Toolkit 11.0+
- PyTorch with CUDA support
- Compatible GPU (compute capability >= 6.0)

## Building

```bash
# Using the build script
./build_kernels.sh

# Or using make
make build-kernels

# Or directly with Python
python setup_kernels.py build_ext --inplace
```

## Integration

The CUDA kernels integrate with the Python codebase through PyTorch extensions and are automatically used when:

- CUDA is available
- Input tensors are on GPU
- Performance optimization is enabled