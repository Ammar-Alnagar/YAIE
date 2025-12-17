# Build System and CUDA Kernels

## Overview

The Mini-YAIE project includes a comprehensive build system for compiling custom CUDA kernels that provide optimized performance for SGLang-style inference operations. The build system is designed to be both educational and production-ready, allowing users to learn CUDA kernel development while achieving high performance.

## Build System Architecture

### Build Scripts

The project provides multiple ways to build kernels:

#### Shell Script: `build_kernels.sh`
```bash
#!/bin/bash
# Comprehensive build script for CUDA kernels
./build_kernels.sh
```

#### Makefile Integration
```makefile
# Build kernels using make
make build-kernels
```

#### Direct Python Build
```bash
# Direct build using Python setup
python setup_kernels.py build_ext --inplace
```

### Build Dependencies

The build system requires:

- **CUDA Toolkit**: Version 11.0 or higher
- **PyTorch with CUDA Support**: For CUDA extensions
- **NVIDIA GPU**: With compute capability >= 6.0
- **System Compiler**: GCC/Clang with C++14 support
- **Python Development Headers**: For Python C API

## CUDA Kernel Design

### SGLang-Style Optimization Focus

The CUDA kernels are specifically designed to optimize SGLang-style inference operations:

1. **Radix Tree Operations**: Efficient prefix matching on GPU
2. **Paged Attention**: Optimized attention for paged KV-cache
3. **Memory Operations**: High-performance memory management
4. **Radix Attention**: GPU-optimized radial attention with prefix sharing

### Performance Goals

The kernels target SGLang-specific optimizations:

- **Memory Bandwidth Optimization**: Minimize memory access overhead
- **Computation Sharing**: Implement efficient prefix sharing on GPU
- **Batch Processing**: Optimize for variable-length batch processing
- **Cache Efficiency**: Optimize for GPU cache hierarchies

## CUDA Kernel Components

### 1. Memory Operations Kernels

#### Paged KV-Cache Operations
```cuda
// Efficient operations on paged key-value cache
__global__ void copy_blocks_kernel(
    float* dst_keys, float* dst_values,
    float* src_keys, float* src_values,
    int* block_mapping, int num_blocks
);
```

#### Block Management
- Block allocation and deallocation
- Memory copying between blocks
- Block state management

### 2. Flash Attention Kernels

#### Optimized Attention Computation
```cuda
// Optimized attention for both prefill and decode phases
template<typename T>
__global__ void flash_attention_kernel(
    const T* q, const T* k, const T* v,
    T* output, float* lse,  // logsumexp for numerical stability
    int num_heads, int head_dim, int seq_len
);
```

#### Features
- Memory-efficient attention computation
- Numerical stability with logsumexp
- Support for variable sequence lengths
- Optimized memory access patterns

### 3. Paged Attention Kernels

#### Paged Memory Access
```cuda
// Attention with paged key-value cache support
__global__ void paged_attention_kernel(
    float* output,
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const int* block_tables,
    const int* context_lens,
    int num_kv_heads, int head_dim, int block_size
);
```

#### Features
- Direct paged cache access patterns
- Efficient block index computation
- Memory coalescing optimization
- Support for shared prefix computation

### 4. Radix Operations Kernels

#### Prefix Matching Operations
```cuda
// GPU-accelerated prefix matching for SGLang-style sharing
__global__ void radix_tree_lookup_kernel(
    int* token_ids, int* request_ids,
    int* prefix_matches, int batch_size
);
```

#### Features
- Parallel prefix matching
- Efficient tree traversal on GPU
- Shared computation identification
- Batch processing optimization

## Build Process Details

### Setup Configuration

The build system uses PyTorch's `setup.py` for CUDA extension compilation:

```python
# setup_kernels.py
from torch.utils.cpp_extension import setup, CUDAExtension

setup(
    name="yaie_kernels",
    ext_modules=[
        CUDAExtension(
            name="yaie_kernels",
            sources=[
                "kernels/cuda/radix_attention.cu",
                "kernels/cuda/paged_attention.cu",
                "kernels/cuda/memory_ops.cu",
                "kernels/cuda/radix_ops.cu"
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"]
            }
        )
    ],
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension}
)
```

### Compilation Flags

The build system uses optimization flags for performance:

- `-O3`: Maximum optimization
- `--use_fast_math`: Fast math operations
- `-arch=sm_60`: Target specific GPU architectures
- `-lineinfo`: Include debug line information

### Architecture Targeting

The system supports multiple GPU architectures:

```bash
# Specify target architecture during build
python setup_kernels.py build_ext --inplace --arch=75  # Turing GPUs
python setup_kernels.py build_ext --inplace --arch=80  # Ampere GPUs
```

## CUDA Kernel Implementation Guidelines

### Memory Management

#### Unified Memory vs Regular Memory
```cuda
// Use unified memory for easier management (if available)
cudaMallocManaged(&ptr, size);

// Or regular device memory for better performance
cudaMalloc(&ptr, size);
```

#### Memory Pooling
- Implement memory pooling for frequently allocated objects
- Reuse memory blocks across operations
- Batch memory operations when possible

### Thread Organization

#### Block and Grid Sizing
```cuda
// Optimize for your specific algorithm
dim3 blockSize(256);
dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
```

#### Warp-Level Primitives
- Use warp-level operations for better efficiency
- Align memory accesses with warp boundaries
- Minimize warp divergence

### Synchronization

#### Cooperative Groups
```cuda
#include <cooperative_groups.h>
using namespace cooperative_groups;

// Use cooperative groups for complex synchronization
thread_block block = this_thread_block();
```

#### Memory Barriers
- Use appropriate memory barriers for consistency
- Minimize unnecessary synchronization overhead

## Performance Optimization Strategies

### Memory Bandwidth Optimization

#### Coalesced Access
```cuda
// Ensure memory accesses are coalesced
int tid = blockIdx.x * blockDim.x + threadIdx.x;
// Access data[tid] by threads in order for coalescing
```

#### Shared Memory Usage
- Use shared memory for frequently accessed data
- Implement tiling strategies for large operations
- Minimize global memory access

### Computation Optimization

#### Warp-Level Operations
- Leverage warp-level primitives when possible
- Use vectorized operations (float4, int4)
- Minimize thread divergence within warps

### Kernel Fusion

#### Combined Operations
- Fuse multiple operations into single kernels
- Reduce kernel launch overhead
- Improve memory locality

## Integration with Python

### PyTorch Extensions

The CUDA kernels integrate with PyTorch using extensions:

```python
import torch
import yaie_kernels  # Compiled extension

# Use kernel from Python
result = yaie_kernels.radix_attention_forward(
    query, key, value, 
    radix_tree_info, 
    attention_mask
)
```

### Automatic GPU Management

The integration handles:
- GPU memory allocation
- Device synchronization
- Error propagation
- Backpropagation support

## Error Handling and Debugging

### Build-Time Errors

Common build issues and solutions:

```bash
# CUDA toolkit not found
export CUDA_HOME=/usr/local/cuda

# Architecture mismatch
# Check GPU compute capability and adjust flag

# Missing PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Runtime Error Checking

Kernels should include error checking:

```cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)
```

## Testing and Validation

### Kernel Testing

Test kernels with:
- Unit tests for individual functions
- Integration tests with Python interface
- Performance benchmarks
- Memory correctness validation

### SGLang-Specific Tests

Test SGLang optimization features:
- Prefix sharing correctness
- Memory management validation
- Performance gain verification
- Edge case handling

## Development Workflow

### Iterative Development

The development process includes:

1. **Kernel Design**: Design algorithm for GPU execution
2. **Implementation**: Write CUDA kernel code
3. **Building**: Compile with build system
4. **Testing**: Validate correctness and performance
5. **Optimization**: Profile and optimize based on results

### Profiling and Optimization

Use NVIDIA tools for optimization:

- **Nsight Systems**: Overall system profiling
- **Nsight Compute**: Detailed kernel analysis
- **nvprof**: Legacy profiling tool

## Future Extensions

### Advanced Features

Potential kernel enhancements:
- Quantized attention operations
- Sparse attention kernels
- Custom activation functions
- Advanced memory management

### Hardware Support

Expand support for:
- Different GPU architectures
- Multi-GPU operations
- Heterogeneous computing
- Tensor Core optimizations

This build system and CUDA kernel architecture enables Mini-YAIE to achieve SGLang-style performance optimizations while maintaining educational value and extensibility.