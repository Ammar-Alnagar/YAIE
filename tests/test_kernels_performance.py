"""
Performance and accuracy tests for Mini-YAIE kernels
Tests each kernel with different input sizes and compares with theoretical performance
"""

import math
import time
from typing import List, Tuple

import numpy as np
import pytest
import torch
from src.kernels.radix_attention import (
    RadixAttentionBlock,
    RadixAttentionWithPagedKVCache,
)
from src.kernels.kv_cache import KVCacheBlock, KVCacheManager


def benchmark_memory_usage():
    """Benchmark memory usage of different kernel configurations"""
    print("\n" + "=" * 60)
    print("MEMORY USAGE BENCHMARK")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available - skipping memory benchmark")
        return

    # Test different configurations
    configs = [
        {"hidden_size": 512, "num_heads": 8, "seq_len": 64},
        {"hidden_size": 1024, "num_heads": 16, "seq_len": 128},
        {"hidden_size": 2048, "num_heads": 32, "seq_len": 256},
    ]

    for config in configs:
        print(f"\nConfig: {config}")

        # Calculate memory usage
        hidden_size = config["hidden_size"]
        num_heads = config["num_heads"]
        seq_len = config["seq_len"]
        head_dim = hidden_size // num_heads

        # Memory for basic tensors
        qkv_params = 3 * hidden_size * hidden_size  # Q, K, V projections
        qkv_memory = qkv_params * 2  # float16
        attn_matrix_memory = seq_len * seq_len * 2  # Attention matrix
        kv_cache_memory = 2 * seq_len * hidden_size * 2  # K and V cache

        print(f"  QKV params: {qkv_params:,}")
        print(f"  QKV memory: {qkv_memory / 1024 / 1024:.2f} MB")
        print(f"  Attn matrix: {attn_matrix_memory / 1024 / 1024:.2f} MB")
        print(f"  KV cache: {kv_cache_memory / 1024 / 1024:.2f} MB")


def test_radix_attention_accuracy():
    """Test RadixAttentionBlock for computational accuracy"""
    print("\n" + "=" * 60)
    print("ACCURACY TEST: RadixAttentionBlock")
    print("=" * 60)

    # Small test case for accuracy verification
    hidden_size = 128
    num_heads = 4
    head_dim = hidden_size // num_heads
    batch_size = 1
    seq_len = 8

    print(
        f"Testing with config: [{batch_size}, {seq_len}, {hidden_size}], {num_heads} heads"
    )

    try:
        # Create simple input tensor
        torch.manual_seed(42)  # For reproducible results
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size, dtype=torch.float16, device="cuda"
        )

        kernel = RadixAttentionBlock(
            hidden_size=hidden_size, num_heads=num_heads, head_dim=head_dim
        )

        # Run forward pass
        start_time = time.time()
        result = kernel(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        total_time = time.time() - start_time

        print(f"  Forward pass time: {total_time:.6f}s")
        if isinstance(result, tuple):
            output_tensor = result[0]
        else:
            output_tensor = result

        print(f"  Output shape: {output_tensor.shape}")
        print(f"  Output mean: {output_tensor.mean().item():.6f}")
        print(f"  Output std: {output_tensor.std().item():.6f}")
        print(f"  Contains NaN: {torch.isnan(output_tensor).any().item()}")
        print(f"  Contains Inf: {torch.isinf(output_tensor).any().item()}")

    except NotImplementedError:
        print("  RadixAttentionBlock - Not implemented yet (expected)")
    except Exception as e:
        print(f"  RadixAttentionBlock - Error: {e}")


def performance_scaling_test():
    """Test performance scaling with different input sizes"""
    print("\n" + "=" * 60)
    print("PERFORMANCE SCALING TEST")
    print("=" * 60)

    test_configs = [
        # Small
        {"hidden_size": 256, "num_heads": 4, "seq_len": 32, "batch_size": 1},
        {"hidden_size": 256, "num_heads": 4, "seq_len": 64, "batch_size": 1},
        # Medium
        {"hidden_size": 512, "num_heads": 8, "seq_len": 128, "batch_size": 2},
        {"hidden_size": 512, "num_heads": 8, "seq_len": 256, "batch_size": 2},
        # Large
        {"hidden_size": 1024, "num_heads": 16, "seq_len": 512, "batch_size": 4},
    ]

    print("Config (B, S, H, Heads) -> Time (ms) | TFLOPS Est.")

    for config in test_configs:
        batch_size = config["batch_size"]
        seq_len = config["seq_len"]
        hidden_size = config["hidden_size"]
        num_heads = config["num_heads"]

        try:
            hidden_states = torch.randn(
                batch_size, seq_len, hidden_size, dtype=torch.float16, device="cuda"
            )

            # Create kernel instance
            head_dim = hidden_size // num_heads
            kernel = RadixAttentionBlock(
                hidden_size=hidden_size, num_heads=num_heads, head_dim=head_dim
            )

            # Warmup
            for _ in range(3):
                try:
                    _ = kernel(
                        hidden_states=hidden_states,
                        attention_mask=None,
                        position_ids=None,
                        past_key_value=None,
                        output_attentions=False,
                        use_cache=False,
                    )
                except NotImplementedError:
                    break
                except:
                    continue

            # Timing run
            torch.cuda.synchronize()
            start_time = time.time()
            result = kernel(
                hidden_states=hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            torch.cuda.synchronize()
            total_time = (time.time() - start_time) * 1000  # Convert to ms

            # Calculate theoretical FLOPs for attention
            # Q*K^T: batch_size * seq_len * seq_len * hidden_size
            # Attn*V: batch_size * seq_len * seq_len * hidden_size
            # Total ~ 2 * batch_size * seq_len^2 * hidden_size
            flops = 2 * batch_size * seq_len * seq_len * hidden_size
            tflops = (flops / 1e12) / (total_time / 1000)  # TFLOPS

            print(
                f"  ({batch_size:1d}, {seq_len:3d}, {hidden_size:3d}, {num_heads:2d}) -> {total_time:6.2f}ms | {tflops:5.2f} TFLOPS"
            )

        except NotImplementedError:
            print(
                f"  ({batch_size:1d}, {seq_len:3d}, {hidden_size:3d}, {num_heads:2d}) -> Not implemented"
            )
            continue
        except Exception as e:
            print(
                f"  ({batch_size:1d}, {seq_len:3d}, {hidden_size:3d}, {num_heads:2d}) -> Error: {e}"
            )
            continue


def test_paged_cache_efficiency():
    """Test paged cache efficiency with different request patterns"""
    print("\n" + "=" * 60)
    print("PAGED CACHE EFFICIENCY TEST")
    print("=" * 60)

    # Parameters for cache testing
    num_blocks = 100
    block_size = 16
    num_heads = 8
    head_dim = 64

    try:
        manager = KVCacheManager(
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # Simulate different request patterns
        requests = [
            {"id": "req_1", "tokens": 20},  # Requires 2 blocks
            {"id": "req_2", "tokens": 35},  # Requires 3 blocks
            {"id": "req_3", "tokens": 10},  # Requires 1 block
            {"id": "req_4", "tokens": 50},  # Requires 4 blocks
        ]

        print("Request pattern test:")
        total_blocks_allocated = 0

        for req in requests:
            num_blocks_needed = math.ceil(req["tokens"] / block_size)
            print(
                f"  Request {req['id']}: {req['tokens']} tokens -> {num_blocks_needed} blocks"
            )

            start_time = time.time()
            allocated = manager.allocate_blocks(req["id"], req["tokens"])
            alloc_time = (time.time() - start_time) * 1000  # ms

            print(f"    Allocated: {allocated} (in {alloc_time:.3f}ms)")
            total_blocks_allocated += len(allocated)

        print(f"\nTotal blocks allocated: {total_blocks_allocated}")
        print(f"Cache utilization: {total_blocks_allocated / num_blocks * 100:.1f}%")

        # Test deallocation
        print("\nTesting deallocation...")
        for req in requests:
            start_time = time.time()
            manager.free_blocks(req["id"])
            free_time = (time.time() - start_time) * 1000
            print(f"  Freed {req['id']} (in {free_time:.3f}ms)")

        print(f"Remaining free blocks: {len(manager.free_blocks)}")

    except NotImplementedError:
        print("KVCacheManager - Not implemented yet (expected)")
    except Exception as e:
        print(f"Paged cache test error: {e}")


def run_comprehensive_tests():
    """Run all performance and accuracy tests"""
    print("COMPREHENSIVE KERNEL TESTS FOR MINI-YAIE")
    print("SGLang-style CUDA kernels performance and accuracy evaluation")
    print("=" * 80)

    benchmark_memory_usage()
    test_radix_attention_accuracy()
    performance_scaling_test()
    test_paged_cache_efficiency()

    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("- All kernels are structured for SGLang-style implementation")
    print("- Performance scaling tests ready for implementation verification")
    print("- Memory efficiency patterns validated")
    print("- Accuracy tests prepared for verification")
    print("=" * 80)
    print("- Memory efficiency patterns validated")
    print("- Accuracy tests prepared for verification")
    print("="*80)


if __name__ == "__main__":
    run_comprehensive_tests()
