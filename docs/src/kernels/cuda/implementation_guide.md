# Kernels Implementation Guide

## Overview

This guide provides a comprehensive walkthrough for implementing the core kernels in Mini-YAIE that enable SGLang-style inference optimization. The implementation focuses on three key areas:

1. **Python Implementations**: Educational implementations of core algorithms
2. **CUDA Kernels**: Performance-optimized GPU implementations  
3. **Integration**: Connecting kernels with the main inference engine

## Implementation Roadmap

### Phase 1: Core Python Kernels

Implement the educational Python versions first:

1. Radix tree for prefix matching
2. Basic attention mechanisms
3. KV-cache management
4. Sampling algorithms

### Phase 2: CUDA Kernel Development

Develop optimized GPU versions:

1. Memory operations kernels
2. Paged attention implementation
3. Flash attention optimization
4. Radix operations acceleration

### Phase 3: Integration and Optimization

Connect kernels to the main system:

1. Engine integration
2. Performance validation
3. Correctness verification

## Python Kernel Implementation

### 1. Radix Tree Implementation

Start with the radix tree that enables prefix sharing:

**File**: `src/kernels/radix_tree.py`

```python
class RadixTreeNode:
    def __init__(self, token_id: Optional[int] = None):
        self.token_id = token_id
        self.children: Dict[int, "RadixTreeNode"] = {}
        self.request_ids: List[str] = []
        self.kv_cache_refs: List[str] = []
        self.is_terminal = False

class RadixTree:
    def __init__(self):
        self.root = RadixTreeNode()
        self.request_to_path: Dict[str, List[int]] = {}
        self.path_to_node: Dict[str, RadixTreeNode] = {}
    
    def insert_request(self, request_id: str, token_ids: List[int]):
        """Insert a request into the radix tree based on its token sequence"""
        current = self.root
        for token_id in token_ids:
            if token_id not in current.children:
                current.children[token_id] = RadixTreeNode(token_id)
            current = current.children[token_id]
            if request_id not in current.request_ids:
                current.request_ids.append(request_id)
        current.is_terminal = True
        self.request_to_path[request_id] = token_ids
        path_str = self._path_to_string(token_ids)
        self.path_to_node[path_str] = current
    
    def find_shared_prefixes(self, token_ids: List[int]) -> Tuple[List[str], int]:
        """Find requests that share prefixes with the given token sequence"""
        current = self.root
        matched_requests = []
        prefix_length = 0
        
        for i, token_id in enumerate(token_ids):
            if token_id in current.children:
                current = current.children[token_id]
                matched_requests.extend(current.request_ids)
                prefix_length = i + 1
            else:
                break
        return list(set(matched_requests)), prefix_length
```

### 2. KV-Cache Management

Implement the paged KV-cache system:

**File**: `src/kernels/kv_cache.py`

```python
class KVCacheBlock:
    def __init__(self, block_id: int, size: int, num_heads: int, head_dim: int, dtype=torch.float16):
        self.block_id = block_id
        self.size = size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.keys = None
        self.values = None
    
    def allocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.keys = torch.zeros(self.size, self.num_heads, self.head_dim, dtype=self.dtype, device=device)
        self.values = torch.zeros(self.size, self.num_heads, self.head_dim, dtype=self.dtype, device=device)

class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int, num_heads: int, head_dim: int, dtype=torch.float16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        
        self.blocks: List[KVCacheBlock] = []
        for i in range(num_blocks):
            block = KVCacheBlock(i, block_size, num_heads, head_dim, dtype)
            self.blocks.append(block)
        
        self.free_block_list: List[int] = list(range(num_blocks))
        self.block_tables: dict = {}
    
    def allocate_blocks(self, request_id: str, num_tokens: int) -> List[int]:
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_block_list) < num_blocks_needed:
            raise RuntimeError(f"Not enough free blocks. Need {num_blocks_needed}, have {len(self.free_block_list)}")
        
        allocated_block_ids = []
        for _ in range(num_blocks_needed):
            block_id = self.free_block_list.pop(0)
            allocated_block_ids.append(block_id)
            self.blocks[block_id].allocate()
        
        self.block_tables[request_id] = allocated_block_ids
        return allocated_block_ids
```

### 3. Radix Attention Implementation

Implement the radial attention mechanism:

**File**: `src/kernels/radix_attention.py`

```python
import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RadixAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, max_position_embeddings: int = 2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        
        total_hidden_dim = num_heads * head_dim
        
        self.q_proj = nn.Linear(hidden_size, total_hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, total_hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, total_hidden_dim, bias=False)
        self.o_proj = nn.Linear(total_hidden_dim, hidden_size, bias=False)
        
        self.register_buffer(
            "cos_cached",
            torch.ones((max_position_embeddings, head_dim), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            torch.ones((max_position_embeddings, head_dim), dtype=torch.float32),
            persistent=False,
        )
        
        self._setup_rope_embeddings()
    
    def _setup_rope_embeddings(self):
        position_ids = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        
        freqs = torch.einsum("i,j->ij", position_ids, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.cos_cached = emb.cos().to(dtype=torch.float16)
        self.sin_cached = emb.sin().to(dtype=torch.float16)
    
    def forward(self, hidden_states: torch.Tensor, position_ids: Optional[torch.Tensor] = None, 
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        cos_to_use = self.cos_cached[:seq_len].to(query.dtype)
        sin_to_use = self.sin_cached[:seq_len].to(query.dtype)
        
        query, key = apply_rotary_pos_emb(query, key, cos_to_use, sin_to_use, position_ids)
        
        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)
        
        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        
        output = self.o_proj(attn_output)
        return output
```

### 4. Sampling Kernel Implementation

Implement the token sampling system:

**File**: `src/kernels/sampling.py`

```python
import torch

class SamplingKernel:
    def sample(self, logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0, top_k: int = -1) -> torch.Tensor:
        if temperature != 1.0:
            logits = logits / temperature
        
        probs = torch.softmax(logits, dim=-1)
        batch_size, vocab_size = probs.shape
        
        if top_k > 0:
            top_k = min(top_k, vocab_size)
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            
            new_probs = torch.zeros_like(probs)
            new_probs.scatter_(1, top_k_indices, top_k_probs)
            new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)
            probs = new_probs
        
        if 0 < top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            mask = cumulative_probs <= top_p
            if mask.shape[-1] > 0:
                mask[..., 0] = True
            
            filtered_probs = torch.zeros_like(probs)
            filtered_probs.scatter_(1, sorted_indices, mask.float() * sorted_probs)
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
            probs = filtered_probs
        
        sampled_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return sampled_ids
```

## CUDA Kernel Implementation

### 1. Memory Operations Kernels

**File**: `src/kernels/cuda/memory_ops.cu`

```cuda
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void copy_blocks_kernel(
    float* key_cache, float* value_cache,
    float* new_key_cache, float* new_value_cache,
    int* block_mapping,  // [src_block_id, dst_block_id] pairs
    int num_heads, int head_dim, int block_size,
    int num_mappings
) {
    int mapping_idx = blockIdx.x;
    if (mapping_idx >= num_mappings) return;
    
    int src_block_id = block_mapping[mapping_idx * 2];
    int dst_block_id = block_mapping[mapping_idx * 2 + 1];
    
    int total_elements_per_block = block_size * num_heads * head_dim;
    
    int tid = threadIdx.x;
    if (tid < total_elements_per_block) {
        int src_idx = src_block_id * total_elements_per_block + tid;
        int dst_idx = dst_block_id * total_elements_per_block + tid;
        
        new_key_cache[dst_idx] = key_cache[src_idx];
        new_value_cache[dst_idx] = value_cache[src_idx];
    }
}

torch::Tensor copy_blocks_cuda(
    torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor block_mapping,
    int num_heads, int head_dim, int block_size
) {
    int num_mappings = block_mapping.size(0);
    
    auto options = key_cache.options();
    auto new_key_cache = torch::zeros_like(key_cache);
    auto new_value_cache = torch::zeros_like(value_cache);
    
    dim3 grid(num_mappings);
    dim3 block(256);  // Use 256 threads per block
    
    copy_blocks_kernel<<<grid, block>>>(
        key_cache.data_ptr<float>(),
        value_cache.data_ptr<float>(),
        new_key_cache.data_ptr<float>(),
        new_value_cache.data_ptr<float>(),
        block_mapping.data_ptr<int>(),
        num_heads, head_dim, block_size,
        num_mappings
    );
    
    cudaDeviceSynchronize();
    return std::make_tuple(new_key_cache, new_value_cache);
}
```

### 2. Paged Attention Kernels

**File**: `src/kernels/cuda/paged_attention.cu`

```cuda
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void paged_attention_kernel(
    float* output,           // [num_seqs, seq_len, num_heads, head_dim]
    const float* query,      // [num_seqs, seq_len, num_heads, head_dim]
    const float* key_cache,  // [num_blocks, block_size, num_kv_heads, head_dim]
    const float* value_cache,// [num_blocks, block_size, num_kv_heads, head_dim]
    const int* block_tables, // [num_seqs, max_blocks_per_seq]
    const int* context_lens, // [num_seqs]
    const int num_kv_heads,
    const int num_queries_per_kv,
    const int head_dim,
    const int block_size,
    const int max_num_blocks_per_seq
) {
    int seq_idx = blockIdx.x;
    int q_head_idx = blockIdx.y;
    int token_idx = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (seq_idx >= gridDim.x || q_head_idx >= gridDim.y || token_idx >= context_lens[seq_idx]) {
        return;
    }
    
    // Get corresponding KV head index
    int kv_head_idx = q_head_idx / num_queries_per_kv;
    
    // Get query vector
    int query_idx = seq_idx * context_lens[seq_idx] * gridDim.y * head_dim +
                    token_idx * gridDim.y * head_dim +
                    q_head_idx * head_dim;
    
    // Shared memory for the current query
    extern __shared__ float shared_mem[];
    float* query_vec = shared_mem;
    
    // Load query vector to shared memory
    for (int d = 0; d < head_dim; d++) {
        query_vec[d] = query[query_idx + d];
    }
    
    // Calculate which block and offset for this token
    int block_idx = token_idx / block_size;
    int block_offset = token_idx % block_size;
    
    // Get physical block number from block table
    int physical_block = block_tables[seq_idx * max_num_blocks_per_seq + block_idx];
    
    // Calculate the actual index in the cache
    int cache_idx = physical_block * block_size * num_kv_heads * head_dim +
                    block_offset * num_kv_heads * head_dim +
                    kv_head_idx * head_dim;
    
    // Perform attention computation
    float sum = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        sum += query_vec[d] * key_cache[cache_idx + d];
    }
    
    // Apply softmax and multiply with value
    float attention_weight = __expf(sum);  // Simplified (real softmax needs normalization)
    for (int d = 0; d < head_dim; d++) {
        int output_idx = seq_idx * context_lens[seq_idx] * gridDim.y * head_dim +
                         token_idx * gridDim.y * head_dim +
                         q_head_idx * head_dim + d;
        output[output_idx] += attention_weight * value_cache[cache_idx + d];
    }
}

torch::Tensor paged_attention_cuda(
    torch::Tensor query, torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor block_tables, torch::Tensor context_lens,
    int num_kv_heads, int num_queries_per_kv
) {
    int num_seqs = query.size(0);
    int seq_len = query.size(1);
    int num_heads = query.size(2);
    int head_dim = query.size(3);
    int block_size = key_cache.size(1);
    int max_blocks_per_seq = block_tables.size(1);
    
    auto output = torch::zeros_like(query);
    
    dim3 grid(num_seqs, num_heads, (seq_len + 255) / 256);  // 256 threads per block
    dim3 block(256);
    
    // Allocate shared memory for query vector
    int shared_mem_size = head_dim * sizeof(float);
    
    paged_attention_kernel<<<grid, block, shared_mem_size>>>(
        output.data_ptr<float>(),
        query.data_ptr<float>(),
        key_cache.data_ptr<float>(),
        value_cache.data_ptr<float>(),
        block_tables.data_ptr<int>(),
        context_lens.data_ptr<int>(),
        num_kv_heads,
        num_queries_per_kv,
        head_dim,
        block_size,
        max_blocks_per_seq
    );
    
    cudaDeviceSynchronize();
    return output;
}
```

### 3. Flash Attention Kernels (Simplified)

**File**: `src/kernels/cuda/flash_attention.cu`

```cuda
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32  // Small tile for educational purposes

__global__ void flash_attention_kernel(
    float* output,
    const float* query,
    const float* key,
    const float* value,
    const int* seq_lens,
    const int num_seqs,
    const int num_heads,
    const int head_dim,
    const int max_seq_len
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    
    if (seq_idx >= num_seqs || head_idx >= num_heads) return;
    
    int current_seq_len = seq_lens[seq_idx];
    
    // Shared memory for tiles
    extern __shared__ float shared_mem[];
    float* s_Q = shared_mem;
    float* s_K = s_Q + TILE_SIZE * head_dim;
    float* s_V = s_K + TILE_SIZE * head_dim;
    float* s_scores = s_V + TILE_SIZE * head_dim;
    
    // Process the sequence in tiles
    for (int q_tile_start = 0; q_tile_start < current_seq_len; q_tile_start += TILE_SIZE) {
        // Load Q tile to shared memory
        for (int i = threadIdx.x; i < TILE_SIZE * head_dim; i += blockDim.x) {
            int q_row = q_tile_start + i / head_dim;
            int q_col = i % head_dim;
            
            if (q_row < current_seq_len) {
                int q_idx = seq_idx * max_seq_len * num_heads * head_dim +
                           q_row * num_heads * head_dim +
                           head_idx * head_dim + q_col;
                s_Q[i] = query[q_idx];
            } else {
                s_Q[i] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // For each K/V tile
        for (int k_tile_start = 0; k_tile_start < current_seq_len; k_tile_start += TILE_SIZE) {
            // Load K and V tiles
            for (int i = threadIdx.x; i < TILE_SIZE * head_dim; i += blockDim.x) {
                int k_row = k_tile_start + i / head_dim;
                int k_col = i % head_dim;
                
                if (k_row < current_seq_len) {
                    int k_idx = seq_idx * max_seq_len * num_heads * head_dim +
                               k_row * num_heads * head_dim +
                               head_idx * head_dim + k_col;
                    s_K[i] = key[k_idx];
                    s_V[i] = value[k_idx];
                } else {
                    s_K[i] = 0.0f;
                    s_V[i] = 0.0f;
                }
            }
            
            __syncthreads();
            
            // Compute attention scores for this tile
            for (int q_local = threadIdx.x; q_local < TILE_SIZE; q_local += blockDim.x) {
                int q_global = q_tile_start + q_local;
                if (q_global >= current_seq_len) continue;
                
                float score_sum = 0.0f;
                float max_score = -INFINITY;
                
                // Compute scores against K tile
                for (int k_local = 0; k_local < TILE_SIZE; k_local++) {
                    int k_global = k_tile_start + k_local;
                    if (k_global >= current_seq_len) continue;
                    
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        int q_offset = q_local * head_dim + d;
                        int k_offset = k_local * head_dim + d;
                        score += s_Q[q_offset] * s_K[k_offset];
                    }
                    
                    // Apply causal mask
                    if (k_global > q_global) score = -INFINITY;
                    
                    // Update max for numerical stability
                    if (score > max_score) max_score = score;
                    
                    s_scores[q_local * TILE_SIZE + k_local] = score;
                }
                
                // Apply softmax with numerical stability
                float exp_sum = 0.0f;
                for (int k_local = 0; k_local < TILE_SIZE; k_local++) {
                    int k_global = k_tile_start + k_local;
                    if (k_global >= current_seq_len || k_global > q_global) {
                        s_scores[q_local * TILE_SIZE + k_local] = 0.0f;
                    } else {
                        float score = s_scores[q_local * TILE_SIZE + k_local];
                        float exp_score = __expf(score - max_score);
                        s_scores[q_local * TILE_SIZE + k_local] = exp_score;
                        exp_sum += exp_score;
                    }
                }
                
                // Normalize scores
                if (exp_sum > 0.0f) {
                    for (int k_local = 0; k_local < TILE_SIZE; k_local++) {
                        s_scores[q_local * TILE_SIZE + k_local] /= exp_sum;
                    }
                }
                
                // Compute output = scores @ V
                for (int d = 0; d < head_dim; d++) {
                    float output_val = 0.0f;
                    for (int k_local = 0; k_local < TILE_SIZE; k_local++) {
                        int v_offset = k_local * head_dim + d;
                        output_val += s_scores[q_local * TILE_SIZE + k_local] * s_V[v_offset];
                    }
                    
                    int out_idx = seq_idx * max_seq_len * num_heads * head_dim +
                                 q_global * num_heads * head_dim +
                                 head_idx * head_dim + d;
                    output[out_idx] = output_val;
                }
            }
            
            __syncthreads();
        }
    }
}

torch::Tensor flash_attention_cuda(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor seq_lens
) {
    int num_seqs = query.size(0);
    int num_heads = query.size(2);
    int head_dim = query.size(3);
    int max_seq_len = query.size(1);
    
    auto output = torch::zeros_like(query);
    
    dim3 grid(num_seqs, num_heads);
    dim3 block(256);
    
    // Shared memory for 3 tiles + scores matrix
    int shared_mem_size = 3 * TILE_SIZE * head_dim * sizeof(float) +
                          TILE_SIZE * TILE_SIZE * sizeof(float);
    
    flash_attention_kernel<<<grid, block, shared_mem_size>>>(
        output.data_ptr<float>(),
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        seq_lens.data_ptr<int>(),
        num_seqs,
        num_heads,
        head_dim,
        max_seq_len
    );
    
    cudaDeviceSynchronize();
    return output;
}
```

## Integration and Testing

### 1. Python-CUDA Interface

Create the Python interface for CUDA kernels:

**File**: `src/kernels/api.py`

```python
"""
API for accessing both Python and CUDA implementations of kernels
"""

import torch
from typing import Optional

# Try to import CUDA extensions
try:
    from . import yaie_kernels  # This will be built from CUDA sources
    CUDA_AVAILABLE = True
except ImportError:
    print("CUDA extensions not available. Using Python implementations.")
    CUDA_AVAILABLE = False

def attention_forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                     use_cuda: bool = True, **kwargs):
    """Unified attention interface that can use CUDA or Python implementation"""
    if CUDA_AVAILABLE and use_cuda and query.is_cuda:
        return yaie_kernels.flash_attention_cuda(query, key, value, 
                                               kwargs.get('seq_lens', None))
    else:
        # Fallback to Python implementation
        from .radix_attention import RadixAttentionBlock
        attention_block = RadixAttentionBlock(
            hidden_size=query.shape[-1],
            num_heads=query.shape[-2],
            head_dim=query.shape[-1] // query.shape[-2]
        )
        return attention_block(query)

def paged_attention_forward(query: torch.Tensor, key_cache: torch.Tensor, 
                           value_cache: torch.Tensor, block_tables: torch.Tensor,
                           context_lens: torch.Tensor, use_cuda: bool = True, **kwargs):
    """Paged attention interface"""
    if CUDA_AVAILABLE and use_cuda and query.is_cuda:
        return yaie_kernels.paged_attention_cuda(
            query, key_cache, value_cache, block_tables, context_lens,
            kwargs.get('num_kv_heads', 1),
            kwargs.get('num_queries_per_kv', 1)
        )
    else:
        # Python fallback would go here
        raise NotImplementedError("Paged attention Python fallback not implemented")

def copy_blocks(key_cache: torch.Tensor, value_cache: torch.Tensor, 
                block_mapping: torch.Tensor, use_cuda: bool = True, **kwargs):
    """Memory copy interface"""
    if CUDA_AVAILABLE and use_cuda and key_cache.is_cuda:
        return yaie_kernels.copy_blocks_cuda(
            key_cache, value_cache, block_mapping,
            kwargs.get('num_heads', 1),
            kwargs.get('head_dim', 64),
            kwargs.get('block_size', 16)
        )
    else:
        # Python fallback would go here
        raise NotImplementedError("Copy blocks Python fallback not implemented")
```

### 2. Testing Framework

Create comprehensive tests:

**File**: `tests/test_kernels.py`

```python
import pytest
import torch
import numpy as np

from src.kernels.radix_tree import RadixTree
from src.kernels.kv_cache import KVCacheManager
from src.kernels.radix_attention import RadixAttentionBlock
from src.kernels.sampling import SamplingKernel
from src.kernels.api import attention_forward, paged_attention_forward, copy_blocks

class TestRadixTree:
    def test_basic_insertion_and_search(self):
        tree = RadixTree()
        
        # Insert requests
        tree.insert_request("req1", [1, 2, 3])
        tree.insert_request("req2", [1, 2, 4])  # Shares prefix [1, 2]
        tree.insert_request("req3", [5, 6, 7])  # No shared prefix
        
        # Test shared prefixes
        shared, length = tree.find_shared_prefixes([1, 2, 5])
        assert "req1" in shared
        assert "req2" in shared
        assert length == 2  # Common prefix [1, 2]
    
    def test_prefix_sharing_graph(self):
        tree = RadixTree()
        tree.insert_request("req1", [1, 2, 3])
        tree.insert_request("req2", [1, 2, 4])
        tree.insert_request("req3", [1, 5, 6])
        
        graph = tree.get_shared_computation_graph()
        # Should show shared computation at token [1]
        assert graph["request_count"] == 3  # All requests start with root
        
class TestKVCacheManager:
    def test_basic_allocation(self):
        cache_manager = KVCacheManager(
            num_blocks=100,
            block_size=16,
            num_heads=8,
            head_dim=64,
            dtype=torch.float16
        )
        
        # Allocate blocks for a request
        blocks = cache_manager.allocate_blocks("req1", 20)  # Need 2 blocks (20/16 = 2)
        assert len(blocks) == 2
        
        # Verify the blocks exist and have proper tensors
        for block_id in blocks:
            block = cache_manager.blocks[block_id]
            assert block.keys is not None
            assert block.values is not None
            assert block.keys.shape == (16, 8, 64)  # block_size, num_heads, head_dim
    
    def test_block_reuse(self):
        cache_manager = KVCacheManager(
            num_blocks=10,
            block_size=16,
            num_heads=8,
            head_dim=64
        )
        
        # Allocate all blocks
        req_ids = [f"req{i}" for i in range(10)]
        for req_id in req_ids:
            cache_manager.allocate_blocks(req_id, 10)
        
        assert len(cache_manager.free_block_list) == 0
        
        # Free some blocks
        cache_manager.free_blocks("req0")
        cache_manager.free_blocks("req1")
        
        assert len(cache_manager.free_block_list) == 2
        assert 0 in cache_manager.free_block_list
        assert 1 in cache_manager.free_block_list

class TestRadixAttention:
    def test_basic_attention_forward(self):
        hidden_size = 512
        num_heads = 8
        head_dim = hidden_size // num_heads
        
        attention = RadixAttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            max_position_embeddings=256
        )
        
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
        
        output = attention(x)
        assert output.shape == (batch_size, seq_len, hidden_size)
        assert not torch.isnan(output).any()
    
    def test_attention_with_past_key_value(self):
        hidden_size = 256
        num_heads = 4
        head_dim = hidden_size // num_heads
        
        attention = RadixAttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim
        )
        
        batch_size = 1
        seq_len = 5
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # First forward pass
        output1, _, past_kv = attention(x, use_cache=True)
        
        # Second forward pass with past key-value
        next_token = torch.randn(batch_size, 1, hidden_size)
        output2, _, _ = attention(next_token, past_key_value=past_kv)
        
        assert output2.shape == (batch_size, 1, hidden_size)

class TestSamplingKernel:
    def test_temperature_sampling(self):
        sampling = SamplingKernel()
        
        # Create logits with one clear winner
        logits = torch.tensor([[10.0, 1.0, 1.0, 1.0]])  # First token is dominant
        
        # High temperature should allow other tokens
        sampled_high_temp = sampling.sample(logits, temperature=2.0)
        assert sampled_high_temp.shape == (1,)
        
        # Low temperature should favor dominant token
        sampled_low_temp = sampling.sample(logits, temperature=0.1)
        assert sampled_low_temp[0] == 0  # Should pick the dominant token
    
    def test_top_p_nucleus_sampling(self):
        sampling = SamplingKernel()
        
        # Create logits where first 3 tokens account for ~90% of probability
        logits = torch.tensor([[2.0, 1.5, 1.0, -10.0, -10.0]])
        
        # Top-p = 0.8 should exclude the last two tokens
        sampled = sampling.sample(logits, top_p=0.8)
        # Should be one of the first 3 tokens
        assert sampled[0] in [0, 1, 2]
    
    def test_top_k_sampling(self):
        sampling = SamplingKernel()
        
        # Create logits with clear ordering
        logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
        
        # Top-k = 2 should only consider first 2 tokens
        sampled = sampling.sample(logits, top_k=2)
        assert sampled[0] in [0, 1]  # Should be one of top 2 tokens

class TestIntegration:
    def test_full_inference_pipeline(self):
        """Test integration of all kernels in a simple pipeline"""
        # This test would simulate a full inference step
        batch_size = 2
        seq_len = 10
        hidden_size = 256
        num_heads = 4
        head_dim = hidden_size // num_heads
        
        # Create input
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # Apply attention
        attention = RadixAttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim
        )
        attn_output = attention(x)
        assert attn_output.shape == (batch_size, seq_len, hidden_size)
        
        # Apply sampling (on logits that would come from LM head)
        logits = torch.randn(batch_size, 1000)  # vocab_size = 1000
        sampling = SamplingKernel()
        sampled_tokens = sampling.sample(logits, temperature=0.7)
        assert sampled_tokens.shape == (batch_size,)

if __name__ == "__main__":
    pytest.main([__file__])
```

## Building and Running

### Setup Configuration

**File**: `setup_kernels.py`

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Check if CUDA is available
def get_extensions():
    extensions = []
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            extensions.append(
                CUDAExtension(
                    name='yaie_kernels',
                    sources=[
                        'src/kernels/cuda/memory_ops.cu',
                        'src/kernels/cuda/paged_attention.cu', 
                        'src/kernels/cuda/flash_attention.cu',
                        'src/kernels/cuda/radix_ops.cu',
                        'src/kernels/cuda/pybind.cpp',  # Python bindings
                    ],
                    extra_compile_args={
                        'cxx': ['-O3'],
                        'nvcc': ['-O3', '--use_fast_math', '-arch=sm_70']
                    }
                )
            )
    except:
        print("CUDA not available, building without CUDA extensions")
    
    return extensions

setup(
    name='yaie_kernels',
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
```

## Performance Optimization Guidelines

### CUDA Optimization Tips

1. **Memory Coalescing**: Ensure threads in a warp access consecutive memory
2. **Shared Memory**: Use for frequently accessed data
3. **Occupancy**: Maximize number of active warps
4. **Reduction Operations**: Use efficient parallel reduction algorithms

### Profiling and Benchmarking

Create benchmarking tools:

```python
import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity

def benchmark_kernel(kernel_func, *args, **kwargs):
    """Benchmark a kernel function"""
    # Warmup
    for _ in range(3):
        result = kernel_func(*args, **kwargs)
    
    # Actual timing
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(10):  # Run multiple times for average
        result = kernel_func(*args, **kwargs)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    return avg_time, result

def profile_kernel(kernel_func, *args, **kwargs):
    """Profile a kernel function"""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        result = kernel_func(*args, **kwargs)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return result
```

This comprehensive implementation guide provides everything needed to implement the core kernels for Mini-YAIE, following SGLang-style optimization principles while maintaining educational value.