"""
Kernels package for Mini-YAIE
Contains all the low-level implementations that need to be filled in
"""

from .cpu_kernels import cpu_attention_forward, cpu_rms_norm, cpu_rope_embeddings
from .cuda_kernels import (
    apply_rope_qk,
    flash_attention_backward,
    flash_attention_forward,
    paged_attention_forward,
    rms_norm,
    silu_and_mul,
)
from .flashinfer import FlashInferAttention, FlashInferConfig, FlashInferPagedAttention
from .kv_cache import KVCacheBlock, KVCacheManager
from .radix_attention import RadixAttentionBlock, RadixAttentionWithPagedKVCache

__all__ = [
    # CUDA kernels
    "flash_attention_forward",
    "flash_attention_backward",
    "paged_attention_forward",
    "apply_rope_qk",
    "rms_norm",
    "silu_and_mul",
    # CPU kernels
    "cpu_attention_forward",
    "cpu_rms_norm",
    "cpu_rope_embeddings",
    # KV-cache management
    "KVCacheManager",
    "KVCacheBlock",
    # Radix attention
    "RadixAttentionBlock",
    "RadixAttentionWithPagedKVCache",
    # FlashInfer
    "FlashInferAttention",
    "FlashInferPagedAttention",
    "FlashInferConfig",
]
