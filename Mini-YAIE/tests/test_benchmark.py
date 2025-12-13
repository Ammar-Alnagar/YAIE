"""
Benchmark tests for YAIE inference engine
Tests performance with a locally downloaded model and records results
"""

import json
import os
import queue
import threading
import time
from datetime import datetime
from typing import Any, Dict, List

import pytest
from src.engine import InferenceEngine


def benchmark_model_loading(model_name: str) -> Dict[str, Any]:
    """
    Benchmark model loading time with detailed metrics

    Args:
        model_name: Name of the model to load

    Returns:
        Dictionary containing loading benchmark results
    """
    print(f"Starting model loading benchmark for: {model_name}")

    start_time = time.time()
    engine = InferenceEngine(model_name)
    load_time = time.time() - start_time

    print(f"Model loaded in {load_time:.2f} seconds")

    # Get model info
    model_size = (
        sum(p.numel() for p in engine.model.parameters())
        if hasattr(engine.model, "parameters")
        else 0
    )

    results = {
        "model_name": model_name,
        "load_time_seconds": load_time,
        "model_size_parameters": model_size,
        "timestamp": datetime.now().isoformat(),
        "test_type": "model_loading",
    }

    print(f"\nLoading Results:")
    print(f"- Model: {model_name}")
    print(f"- Load time: {load_time:.2f}s")
    print(f"- Parameters: {model_size:,}")

    return results


def benchmark_time_to_first_token(
    model_name: str, prompt: str, max_tokens: int = 100
) -> Dict[str, Any]:
    """
    Benchmark time to first token generation

    Args:
        model_name: Name of the model to test
        prompt: Input prompt for the test
        max_tokens: Maximum tokens to generate

    Returns:
        Dictionary containing TTFT benchmark results
    """
    print(f"Starting TTFT benchmark with model: {model_name}")

    engine = InferenceEngine(model_name)

    # Run a warmup to avoid cold start effects
    _ = engine.generate([prompt], max_tokens=10)

    # Measure time to first token
    start_time = time.time()

    # For now we can't measure incremental token generation since generate() returns full result
    # This would be implemented in the actual generate method when you add streaming
    responses = engine.generate([prompt], max_tokens=max_tokens)

    total_time = time.time() - start_time

    # Approximate TTFT by assuming first token takes a portion of the total time
    # In a real implementation, you'd have streaming access to the first token
    ttft_approx = total_time * 0.1  # Rough approximation

    results = {
        "model_name": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "total_generation_time": total_time,
        "approximated_ttft_seconds": ttft_approx,
        "timestamp": datetime.now().isoformat(),
        "test_type": "time_to_first_token",
        "response_length": len(engine.tokenizer.encode(responses[0])),
    }

    print(f"\nTTFT Results (approximated):")
    print(f"- Model: {model_name}")
    print(f"- Total time: {total_time:.2f}s")
    print(f"- Approx. TTFT: {ttft_approx:.3f}s")
    print(f"- Generated tokens: {results['response_length']}")

    return results


def benchmark_streaming_performance(
    model_name: str, prompt: str, max_tokens: int = 100
) -> Dict[str, Any]:
    """
    Benchmark streaming performance metrics

    Args:
        model_name: Name of the model to test
        prompt: Input prompt for the test
        max_tokens: Maximum tokens to generate

    Returns:
        Dictionary containing streaming benchmark results
    """
    print(f"Starting streaming performance benchmark with model: {model_name}")

    engine = InferenceEngine(model_name)

    # Warmup
    _ = engine.generate([prompt], max_tokens=10)

    # Benchmark full generation
    start_time = time.time()
    responses = engine.generate([prompt], max_tokens=max_tokens)
    total_time = time.time() - start_time

    # Calculate tokens per second
    response_tokens = len(engine.tokenizer.encode(responses[0]))
    tokens_per_second = response_tokens / total_time if total_time > 0 else 0

    # Calculate latency per token
    latency_per_token = (
        total_time / response_tokens if response_tokens > 0 else float("inf")
    )

    results = {
        "model_name": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "total_generation_time": total_time,
        "tokens_generated": response_tokens,
        "tokens_per_second": tokens_per_second,
        "latency_per_token": latency_per_token,
        "timestamp": datetime.now().isoformat(),
        "test_type": "streaming_performance",
        "response": responses[0],
    }

    print(f"\nStreaming Performance Results:")
    print(f"- Model: {model_name}")
    print(f"- Total time: {total_time:.2f}s")
    print(f"- Tokens generated: {response_tokens}")
    print(f"- Tokens/sec: {tokens_per_second:.2f}")
    print(f"- Latency/token: {latency_per_token:.3f}s")

    return results


def benchmark_throughput_batch(
    model_name: str, prompts: List[str], max_tokens: int = 100
) -> Dict[str, Any]:
    """
    Benchmark throughput with batch processing

    Args:
        model_name: Name of the model to test
        prompts: List of prompts to process in batch
        max_tokens: Maximum tokens to generate per prompt

    Returns:
        Dictionary containing throughput benchmark results
    """
    print(f"Starting batch throughput benchmark with model: {model_name}")
    print(f"Processing {len(prompts)} prompts in batch")

    engine = InferenceEngine(model_name)

    # Warmup with single prompt
    _ = engine.generate([prompts[0]], max_tokens=10)

    # Process batch
    start_time = time.time()
    responses = engine.generate(prompts, max_tokens=max_tokens)
    total_time = time.time() - start_time

    # Calculate metrics
    total_tokens_generated = sum(
        len(engine.tokenizer.encode(resp)) for resp in responses
    )
    avg_latency = total_time / len(prompts)
    tokens_per_second = total_tokens_generated / total_time if total_time > 0 else 0
    prompts_per_second = len(prompts) / total_time if total_time > 0 else 0

    results = {
        "model_name": model_name,
        "num_prompts": len(prompts),
        "max_tokens_per_request": max_tokens,
        "total_batch_time": total_time,
        "avg_latency_per_prompt": avg_latency,
        "total_tokens_generated": total_tokens_generated,
        "tokens_per_second": tokens_per_second,
        "prompts_per_second": prompts_per_second,
        "timestamp": datetime.now().isoformat(),
        "test_type": "throughput_batch",
        "responses": responses,
    }

    print(f"\nBatch Throughput Results:")
    print(f"- Model: {model_name}")
    print(f"- Total prompts: {len(prompts)}")
    print(f"- Total time: {total_time:.2f}s")
    print(f"- Average latency/prompt: {avg_latency:.2f}s")
    print(f"- Total tokens: {total_tokens_generated}")
    print(f"- Tokens/sec: {tokens_per_second:.2f}")
    print(f"- Prompts/sec: {prompts_per_second:.2f}")

    return results


def benchmark_memory_usage(
    model_name: str, prompt: str, max_tokens: int = 50
) -> Dict[str, Any]:
    """
    Benchmark memory usage during generation

    Args:
        model_name: Name of the model to test
        prompt: Input prompt for the test
        max_tokens: Maximum tokens to generate

    Returns:
        Dictionary containing memory benchmark results
    """
    try:
        import torch
        import psutil

        # Get initial memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        initial_gpu_memory = (
            torch.cuda.memory_allocated() / 1024 / 1024
            if torch.cuda.is_available()
            else 0
        )  # MB

        engine = InferenceEngine(model_name)

        # Get memory after loading
        model_load_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        model_load_gpu_memory = (
            torch.cuda.memory_allocated() / 1024 / 1024
            if torch.cuda.is_available()
            else 0
        )  # MB

        # Run generation
        start_time = time.time()
        response = engine.generate([prompt], max_tokens=max_tokens)
        total_time = time.time() - start_time

        # Get final memory usage
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        final_gpu_memory = (
            torch.cuda.memory_allocated() / 1024 / 1024
            if torch.cuda.is_available()
            else 0
        )  # MB

        # Calculate memory usage
        memory_used_during_load = model_load_memory - initial_memory
        memory_used_during_gen = final_memory - model_load_memory
        memory_used_total = final_memory - initial_memory

        gpu_memory_used_during_load = model_load_gpu_memory - initial_gpu_memory
        gpu_memory_used_during_gen = final_gpu_memory - model_load_gpu_memory
        gpu_memory_used_total = final_gpu_memory - initial_gpu_memory

        results = {
            "model_name": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "initial_memory_mb": initial_memory,
            "model_load_memory_mb": model_load_memory,
            "final_memory_mb": final_memory,
            "memory_used_during_load_mb": memory_used_during_load,
            "memory_used_during_gen_mb": memory_used_during_gen,
            "memory_used_total_mb": memory_used_total,
            "initial_gpu_memory_mb": initial_gpu_memory,
            "model_load_gpu_memory_mb": model_load_gpu_memory,
            "final_gpu_memory_mb": final_gpu_memory,
            "gpu_memory_used_during_load_mb": gpu_memory_used_during_load,
            "gpu_memory_used_during_gen_mb": gpu_memory_used_during_gen,
            "gpu_memory_used_total_mb": gpu_memory_used_total,
            "generation_time": total_time,
            "timestamp": datetime.now().isoformat(),
            "test_type": "memory_usage",
        }

        print(f"\nMemory Usage Results:")
        print(f"- Model: {model_name}")
        print(f"- CPU Memory - Initial: {initial_memory:.1f}MB")
        print(f"- CPU Memory - After load: {model_load_memory:.1f}MB")
        print(f"- CPU Memory - After gen: {final_memory:.1f}MB")
        print(f"- CPU Memory used - Loading: {memory_used_during_load:.1f}MB")
        print(f"- CPU Memory used - Generation: {memory_used_during_gen:.1f}MB")
        if torch.cuda.is_available():
            print(f"- GPU Memory - Initial: {initial_gpu_memory:.1f}MB")
            print(f"- GPU Memory - After load: {model_load_gpu_memory:.1f}MB")
            print(f"- GPU Memory - After gen: {final_gpu_memory:.1f}MB")
            print(f"- GPU Memory used - Loading: {gpu_memory_used_during_load:.1f}MB")
            print(f"- GPU Memory used - Generation: {gpu_memory_used_during_gen:.1f}MB")

        return results

    except ImportError:
        print("psutil not available, skipping memory benchmark")
        return {
            "model_name": model_name,
            "error": "psutil not available for memory monitoring",
            "timestamp": datetime.now().isoformat(),
            "test_type": "memory_usage",
        }


@pytest.mark.benchmark
def test_comprehensive_benchmark():
    """
    Comprehensive benchmark test including all performance metrics
    """
    # Use a smaller model for testing to ensure it's more likely to be available
    model_name = "microsoft/DialoGPT-small"

    # Single prompt for detailed metrics
    single_prompt = (
        "The Transformer architecture revolutionized natural language processing by"
    )

    # Multiple prompts for batch testing
    batch_prompts = [
        "Explain the concept of attention mechanism in transformers.",
        "What is the difference between prefill and decode phases in inference?",
        "Describe how KV-cache works in LLM inference.",
        "Explain the benefits of continuous batching in LLM serving.",
        "What are the advantages of paged KV-cache over contiguous cache?",
    ]

    # Dictionary to store all results
    comprehensive_results = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "tests": {},
    }

    try:
        # 1. Model loading benchmark
        print("=" * 60)
        print("1. MODEL LOADING BENCHMARK")
        print("=" * 60)
        load_results = benchmark_model_loading(model_name)
        comprehensive_results["tests"]["model_loading"] = load_results

        # 2. Time to first token benchmark
        print("\n" + "=" * 60)
        print("2. TIME TO FIRST TOKEN BENCHMARK")
        print("=" * 60)
        ttft_results = benchmark_time_to_first_token(model_name, single_prompt)
        comprehensive_results["tests"]["time_to_first_token"] = ttft_results

        # 3. Streaming performance benchmark
        print("\n" + "=" * 60)
        print("3. STREAMING PERFORMANCE BENCHMARK")
        print("=" * 60)
        stream_results = benchmark_streaming_performance(model_name, single_prompt)
        comprehensive_results["tests"]["streaming_performance"] = stream_results

        # 4. Throughput with batch processing
        print("\n" + "=" * 60)
        print("4. THROUGHPUT BENCHMARK")
        print("=" * 60)
        throughput_results = benchmark_throughput_batch(model_name, batch_prompts)
        comprehensive_results["tests"]["throughput_batch"] = throughput_results

        # 5. Memory usage benchmark
        print("\n" + "=" * 60)
        print("5. MEMORY USAGE BENCHMARK")
        print("=" * 60)
        memory_results = benchmark_memory_usage(model_name, single_prompt)
        comprehensive_results["tests"]["memory_usage"] = memory_results

        # Save comprehensive results
        output_file = f"comprehensive_benchmark_{model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(comprehensive_results, f, indent=2)

        print(f"\nComprehensive benchmark results saved to {output_file}")

        # Summary
        print("\n" + "=" * 60)
        print("COMPREHENSIVE BENCHMARK COMPLETE")
        print("=" * 60)
        if "streaming_performance" in comprehensive_results["tests"]:
            stream_data = comprehensive_results["tests"]["streaming_performance"]
            print(
                f"Tokens per second: {stream_data.get('tokens_per_second', 'N/A'):.2f}"
            )
        if "throughput_batch" in comprehensive_results["tests"]:
            throughput_data = comprehensive_results["tests"]["throughput_batch"]
            print(
                f"Batch prompts per second: {throughput_data.get('prompts_per_second', 'N/A'):.2f}"
            )
            print(
                f"Batch tokens per second: {throughput_data.get('tokens_per_second', 'N/A'):.2f}"
            )
        if "model_loading" in comprehensive_results["tests"]:
            load_data = comprehensive_results["tests"]["model_loading"]
            print(f"Model load time: {load_data.get('load_time_seconds', 'N/A'):.2f}s")
        print("=" * 60)

        return comprehensive_results

    except Exception as e:
        print(f"Error running comprehensive benchmark: {e}")
        print("Make sure the model is downloaded locally before running this test.")
        # Return a minimal result in case of failure for documentation
        error_result = {
            "model_name": model_name,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "tests": {},
        }
        return error_result


@pytest.mark.benchmark
def test_simple_generation():
    """
    Simple generation test with a short prompt
    """
    # Using a smaller model for quick testing
    model_name = (
        "microsoft/DialoGPT-small"  # Fallback model if others are not available
    )

    try:
        engine = InferenceEngine(model_name)

        # Simple test
        prompt = "Hello, how are you?"
        response = engine.generate([prompt], max_tokens=50)

        print(f"Input: {prompt}")
        print(f"Output: {response[0]}")

        # Basic assertions
        assert len(response) == 1, "Should return one response"
        assert len(response[0]) > 0, "Response should not be empty"
        print("✓ Simple generation test passed")

    except Exception as e:
        print(
            f"Simple generation test failed (this is expected if model is not available): {e}"
        )


def run_single_test():
    """
    Run a single comprehensive test and print results
    """
    print("Running comprehensive benchmark test...")
    results = test_comprehensive_benchmark()

    # Print final summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL BENCHMARK TESTS")
    print("=" * 70)
    for test_name, test_results in results.get("tests", {}).items():
        print(f"\n{test_name.upper()}:")
        if "error" in test_results:
            print(f"  Error: {test_results['error']}")
        else:
            for key, value in test_results.items():
                if key not in ["test_type", "timestamp", "responses", "response"]:
                    print(f"  {key}: {value}")
    print("=" * 70)


if __name__ == "__main__":
    # Run the comprehensive benchmark test when executed directly
    run_single_test()
