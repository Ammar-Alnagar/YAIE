# YAIE Kernels Implementation Guide

This document outlines all the kernels that need to be implemented in the YAIE inference engine. Each section describes what the kernel does, why it's important, and implementation guidance.

## 1. Attention Kernels

### 1.1 FlashAttention Forward Pass
- **File**: `src/kernels/cuda_kernels.py`
- **Function**: `flash_attention_forward(q, k, v, causal_mask=None)`
- **Purpose**: Efficient attention computation with reduced memory complexity
- **Implementation Notes**:
  - Standard attention has O(n²) memory complexity, FlashAttention has O(n)
  - Uses tiling and recalculation to avoid storing large attention matrices
  - Critical for processing long sequences efficiently
- **Prerequisites**: Understanding of attention mechanism and CUDA programming

### 1.2 FlashAttention Backward Pass
- **File**: `src/kernels/cuda_kernels.py`
- **Function**: `flash_attention_backward(dout, q, k, v, output)`
- **Purpose**: Gradient computation for FlashAttention (needed for training, optional for inference)
- **Implementation Notes**:
  - Must mirror the forward pass computation
  - Handle gradient flow properly
- **Note**: For pure inference, this may not be required

### 1.3 Paged Attention
- **File**: `src/kernels/cuda_kernels.py`
- **Function**: `paged_attention_forward(q, k_cache, v_cache, block_tables, context_lens, max_context_len)`
- **Purpose**: Efficient KV-cache management using paging
- **Implementation Notes**:
  - Allows variable-length sequences in the same batch
  - Reduces memory fragmentation compared to contiguous cache
  - Key to supporting continuous batching efficiently
  - Uses block tables to map sequence positions to memory blocks

## 2. Radix Attention Kernels

### 2.1 Radix Attention Block
- **File**: `src/kernels/radix_attention.py`
- **Class**: `RadixAttentionBlock`
- **Purpose**: Implementation of radial attention with prefix sharing inspired by SGLang
- **Implementation Notes**:
  - Efficiently shares computation for requests with common prefixes
  - Key to achieving high throughput in continuous batching
  - Should integrate with paged KV-cache for memory efficiency
  - Must handle varying sequence lengths in batches

### 2.2 Radix Attention with Paged KV-Cache
- **File**: `src/kernels/radix_attention.py`
- **Class**: `RadixAttentionWithPagedKVCache`
- **Purpose**: Radial attention with integrated paged KV-cache management
- **Implementation Notes**:
  - Combines radial attention with memory-efficient paging
  - Manages block allocation for different requests
  - Handles block sharing between requests with common prefixes
  - Critical for memory-efficient prefix sharing

## 3. FlashInfer-Style Kernels

### 3.1 FlashInfer Attention
- **File**: `src/kernels/flashinfer.py`
- **Class**: `FlashInferAttention`
- **Purpose**: Optimized attention implementation following FlashInfer's approach
- **Implementation Notes**:
  - Optimized differently for prefill and decode phases
  - Focuses on memory bandwidth efficiency
  - Designed for both dense and potentially sparse attention patterns
  - High-performance implementation for generation workloads

### 3.2 FlashInfer Paged Attention
- **File**: `src/kernels/flashinfer.py`
- **Class**: `FlashInferPagedAttention`
- **Purpose**: Paged attention optimized following FlashInfer's methodology
- **Implementation Notes**:
  - Memory layout optimized for GPU access patterns
  - Efficient page lookup mechanisms
  - Specialized for variable-length sequence handling
  - Optimized for batched inference scenarios

## 4. Position Embedding Kernels

### 4.1 RoPE (Rotary Position Embedding)
- **File**: `src/kernels/cuda_kernels.py`
- **Function**: `apply_rope_qk(q, k, cos, sin, position_ids)`
- **Purpose**: Apply rotary embeddings to query and key tensors
- **Implementation Notes**:
  - Many modern models (LLaMA, etc.) use RoPE instead of absolute position embeddings
  - Involves rotating query and key vectors based on position
  - Critical for understanding token positions/relationships

## 5. Normalization Kernels

### 5.1 RMS Normalization
- **File**: `src/kernels/cuda_kernels.py`
- **Function**: `rms_norm(input_tensor, weight, epsilon=1e-6)`
- **Purpose**: Root Mean Square normalization used in many transformer models
- **Implementation Notes**:
  - More efficient than LayerNorm
  - Common in models like LLaMA
  - Requires computing RMS and applying learnable weights

## 6. Activation Kernels

### 6.1 SiLU and Multiplication (SwiGLU component)
- **File**: `src/kernels/cuda_kernels.py`
- **Function**: `silu_and_mul(input_tensor)`
- **Purpose**: Fused activation function used in SwiGLU (in some models)
- **Implementation Notes**:
  - Separates input into two halves: applies SiLU to first half
  - Multiplies first half with second half
  - Fused kernel is more efficient than separate operations

## 7. CPU Fallback Kernels

### 7.1 CPU Attention
- **File**: `src/kernels/cpu_kernels.py`
- **Function**: `cpu_attention_forward(q, k, v, mask=None)`
- **Purpose**: CPU implementation of attention for fallback when GPU unavailable
- **Implementation Notes**:
  - Simpler implementation than FlashAttention
  - Standard matrix operations with attention masking
  - Optimized for CPU cache efficiency

### 7.2 CPU RMS Normalization
- **File**: `src/kernels/cpu_kernels.py`
- **Function**: `cpu_rms_norm(input_tensor, weight, epsilon=1e-6)`
- **Purpose**: CPU implementation of RMS normalization
- **Implementation Notes**:
  - Same algorithm as GPU version but optimized for CPU
  - Consider SIMD instructions for better performance

### 7.3 CPU RoPE
- **File**: `src/kernels/cpu_kernels.py`
- **Function**: `cpu_rope_embeddings(input_tensor, cos, sin, position_ids)`
- **Purpose**: CPU implementation of RoPE
- **Implementation Notes**:
  - Same algorithm as GPU version but with CPU optimizations

## 8. KV-Cache Management

### 8.1 KVCacheBlock
- **File**: `src/kernels/kv_cache.py`
- **Class**: `KVCacheBlock`
- **Purpose**: Represents a single memory block in the paged KV-cache
- **Implementation Notes**:
  - Manages allocation and deallocation of GPU memory
  - Stores key and value tensors for a fixed number of tokens
  - Must handle different data types (float16, bfloat16, etc.)

### 8.2 KVCacheManager
- **File**: `src/kernels/kv_cache.py`
- **Class**: `KVCacheManager`
- **Purpose**: Manages the entire paged KV-cache system
- **Implementation Notes**:
  - Tracks which blocks are free/allocated
  - Manages block allocation/deallocation for requests
  - Handles block copying for operations like beam search
  - Implements memory management policies

## 9. Memory Management Kernels (Advanced)

### 9.1 Memory Pool Management
- **File**: `src/kernels/cuda_kernels.py`
- **Function**: `memory_pool_allocate(size)`, `memory_pool_free(ptr)`
- **Purpose**: Efficient GPU memory allocation/deallocation
- **Implementation Notes**:
  - Avoids costly CUDA malloc/free operations during inference
  - Pre-allocates large chunks of GPU memory
  - Manages internal bookkeeping of available memory
- **Note**: Implement after basic kernels working

### 9.2 Block Swapping (For Memory Constraints)
- **File**: `src/kernels/kv_cache.py`
- **Function**: `swap_blocks_to_cpu(block_ids)`, `swap_blocks_from_cpu(block_ids)`
- **Purpose**: Move blocks between GPU and CPU when GPU memory is full
- **Implementation Notes**:
  - Advanced feature for systems with limited GPU memory
  - Introduces latency but allows processing longer contexts
- **Note**: Implement after basic paged attention working

## Implementation Order Recommendation

1. **Start with basic kernels**:
   - CPU implementations first (easier to debug)
   - RMS normalization
   - Basic attention
   - RoPE

2. **Move to GPU implementations**:
   - RMS normalization (GPU)
   - RoPE (GPU)
   - Basic attention (GPU)

3. **Implement KV-cache management**:
   - KVCacheBlock
   - KVCacheManager
   - Basic paged attention

4. **Add radial and FlashInfer attention**:
   - RadixAttentionBlock
   - RadixAttentionWithPagedKVCache
   - FlashInferAttention
   - FlashInferPagedAttention

5. **Optimize with advanced kernels**:
   - FlashAttention
   - SiLU and Mul fusion
   - Memory pooling

6. **Advanced features** (optional):
   - Block swapping
   - More complex attention variants

## Testing Strategy

1. **Unit tests for each kernel** with simple inputs and expected outputs
2. **Compare results** against PyTorch implementations to verify correctness
3. **Benchmark performance** to ensure efficiency improvements
4. **Integration tests** to ensure kernels work together in the full inference pipeline

## Resources for Implementation

- FlashAttention paper: https://arxiv.org/abs/2205.14135
- PagedAttention paper: https://arxiv.org/abs/2309.06180 (vLLM)
- SGLang paper: https://arxiv.org/abs/2308.07561 (RadixAttention)
- FlashInfer: https://github.com/flashinfer-ai/flashinfer
- CUDA programming guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/
