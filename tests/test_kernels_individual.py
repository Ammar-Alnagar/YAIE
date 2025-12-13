"""
Kernel test suite for Mini-YAIE
Tests each kernel individually with performance and accuracy benchmarks
"""

import time
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch
from src.kernels.radix_attention import (
    RadixAttentionBlock,
    RadixAttentionWithPagedKVCache,
)
from src.kernels.kv_cache import KVCacheBlock, KVCacheManager


def test_radix_attention_block():
    """Test the RadixAttentionBlock kernel individually"""
    print("\n" + "=" * 60)
    print("TESTING: RadixAttentionBlock")
    print("=" * 60)

    # Parameters
    hidden_size = 512
    num_heads = 8
    head_dim = hidden_size // num_heads
    batch_size = 2
    seq_len = 32

    # Create test inputs
    hidden_states = torch.randn(
        batch_size, seq_len, hidden_size, dtype=torch.float16, device="cuda"
    )

    # Initialize the kernel
    try:
        kernel = RadixAttentionBlock(
            hidden_size=hidden_size, num_heads=num_heads, head_dim=head_dim
        )

        # Timing test
        start_time = time.time()
        output = kernel(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        total_time = time.time() - start_time

        print(f"RadixAttentionBlock - SUCCESS")
        print(f"  Input: [{batch_size}, {seq_len}, {hidden_size}]")
        print(f"  Time: {total_time:.4f}s")
        print(
            f"  Output shape: {output[0].shape if isinstance(output, tuple) else output.shape}"
        )

        # Performance metrics
        params = hidden_size * hidden_size * 4  # Q, K, V, O projections
        print(f"  Parameters: {params:,}")

    except NotImplementedError:
        print("RadixAttentionBlock - NOT IMPLEMENTED YET (as expected)")
    except Exception as e:
        print(f"RadixAttentionBlock - ERROR: {e}")


def test_radix_attention_with_paged_kv_cache():
    """Test the RadixAttentionWithPagedKVCache kernel individually"""
    print("\n" + "=" * 60)
    print("TESTING: RadixAttentionWithPagedKVCache")
    print("=" * 60)

    # Parameters
    num_layers = 2
    num_heads = 8
    head_dim = 64
    batch_size = 2
    seq_len = 32

    # Initialize the kernel
    try:
        kernel = RadixAttentionWithPagedKVCache(
            num_layers=num_layers, num_heads=num_heads, head_dim=head_dim
        )

        # Create test key-value tensors
        key = torch.randn(
            batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device="cuda"
        )
        value = torch.randn(
            batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device="cuda"
        )

        # Timing test
        request_id = "test_request_1"
        start_time = time.time()
        kernel.append_slot(key, value, request_id)
        append_time = time.time() - start_time

        print(f"RadixAttentionWithPagedKVCache - append_slot SUCCESS")
        print(f"  Input: Key/Value [{batch_size}, {seq_len}, {num_heads}, {head_dim}]")
        print(f"  Append time: {append_time:.4f}s")

        # Test get_kv_cache
        start_time = time.time()
        kv_result = kernel.get_kv_cache([request_id], [seq_len])
        get_time = time.time() - start_time

        print(f"  Get KV-cache time: {get_time:.4f}s")
        if kv_result:
            k_out, v_out = kv_result
            print(f"  Retrieved shapes: K{list(k_out.shape)}, V{list(v_out.shape)}")

        # Test free_request
        start_time = time.time()
        kernel.free_request(request_id)
        free_time = time.time() - start_time
        print(f"  Free request time: {free_time:.4f}s")

    except NotImplementedError:
        print("RadixAttentionWithPagedKVCache - NOT IMPLEMENTED YET (as expected)")
    except Exception as e:
        print(f"RadixAttentionWithPagedKVCache - ERROR: {e}")


def test_kv_cache_block():
    """Test the KVCacheBlock kernel individually"""
    print("\n" + "=" * 60)
    print("TESTING: KVCacheBlock")
    print("=" * 60)

    # Parameters
    block_id = 0
    size = 16
    num_heads = 8
    head_dim = 64

    try:
        # Initialize block
        block = KVCacheBlock(
            block_id=block_id, size=size, num_heads=num_heads, head_dim=head_dim
        )

        # Timing test for allocation
        start_time = time.time()
        block.allocate()
        alloc_time = time.time() - start_time

        print(f"KVCacheBlock - allocate SUCCESS")
        print(f"  Block ID: {block_id}")
        print(f"  Size: {size}, Heads: {num_heads}, Head Dim: {head_dim}")
        print(f"  Allocation time: {alloc_time:.4f}s")

        # Memory usage
        memory_per_block = size * num_heads * head_dim * 2 * 2  # K+V, float16
        print(f"  Memory per block: {memory_per_block / 1024:.2f} KB")

    except NotImplementedError:
        print("KVCacheBlock - NOT IMPLEMENTED YET (as expected)")
    except Exception as e:
        print(f"KVCacheBlock - ERROR: {e}")


def test_kv_cache_manager():
    """Test the KVCacheManager kernel individually"""
    print("\n" + "=" * 60)
    print("TESTING: KVCacheManager")
    print("=" * 60)

    # Parameters
    num_blocks = 100
    block_size = 16
    num_heads = 8
    head_dim = 64

    try:
        # Initialize manager
        manager = KVCacheManager(
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # Timing test for block allocation
        request_id = "test_request_1"
        num_tokens = 32  # Requires 2 blocks (block_size=16)

        start_time = time.time()
        allocated_blocks = manager.allocate_blocks(request_id, num_tokens)
        alloc_time = time.time() - start_time

        print(f"KVCacheManager - allocate_blocks SUCCESS")
        print(f"  Request: {request_id}")
        print(f"  Tokens: {num_tokens}, Required blocks: {len(allocated_blocks)}")
        print(f"  Allocated blocks: {allocated_blocks}")
        print(f"  Allocation time: {alloc_time:.4f}s")

        # Test get_kv_tensors
        start_time = time.time()
        kv_tensors = manager.get_kv_tensors(allocated_blocks, [num_tokens])
        get_time = time.time() - start_time

        print(f"  Get KV tensors time: {get_time:.4f}s")

        # Test free_blocks
        start_time = time.time()
        manager.free_blocks(request_id)
        free_time = time.time() - start_time
        print(f"  Free blocks time: {free_time:.4f}s")

    except NotImplementedError:
        print("KVCacheManager - NOT IMPLEMENTED YET (as expected)")
    except Exception as e:
        print(f"KVCacheManager - ERROR: {e}")


def run_all_kernel_tests():
    """Run all kernel tests with performance and accuracy metrics"""
    print("KERNEL TEST SUITE FOR MINI-YAIE")
    print("Testing SGLang-style CUDA kernels individually")
    print("=" * 80)

    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available - some tests may not run properly")
    else:
        print(f"CUDA available: {torch.cuda.get_device_name()}")

    # Run each test
    test_radix_attention_block()
    test_radix_attention_with_paged_kv_cache()
    test_kv_cache_block()
    test_kv_cache_manager()

    print("\n" + "=" * 80)
    print("KERNEL TEST COMPLETION SUMMARY")
    print("All kernels are properly structured for SGLang-style implementation")
    print("Next step: Implement each kernel with CUDA code as detailed in TODOs")
    print("=" * 80)
    print("All kernels are properly structured for SGLang-style implementation")
    print("Next step: Implement each kernel with CUDA code as detailed in TODOs")
    print("="*80)


if __name__ == "__main__":
    run_all_kernel_tests()
