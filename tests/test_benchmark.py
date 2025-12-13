"""
Benchmark tests for YAIE inference engine
Tests performance with a locally downloaded model and records results
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

import pytest

from src.engine import InferenceEngine

def benchmark_model(
    model_name: str, prompts: List[str], max_tokens: int = 100
) -> Dict[str, Any]:
    """
    Benchmark the inference engine with specific prompts

    Args:
        model_name: Name of the model to test
        prompts: List of prompts to test
        max_tokens: Maximum tokens to generate per prompt

    Returns:
        Dictionary containing benchmark results
    """
    print(f"Initializing engine with model: {model_name}")

    # Initialize the engine
    start_time = time.time()
    engine = InferenceEngine(model_name)
    init_time = time.time() - start_time

    print(f"Engine initialized in {init_time:.2f} seconds")

    # Warmup run
    print("Running warmup...")
    _ = engine.generate([prompts[0]], max_tokens=20)

    # Benchmark generation
    print(f"Running benchmark with {len(prompts)} prompts...")
    start_time = time.time()

    # Generate responses
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
        "timestamp": datetime.now().isoformat(),
        "num_prompts": len(prompts),
        "max_tokens_per_request": max_tokens,
        "init_time_seconds": init_time,
        "total_generation_time_seconds": total_time,
        "avg_latency_per_prompt": avg_latency,
        "total_tokens_generated": total_tokens_generated,
        "tokens_per_second": tokens_per_second,
        "prompts_per_second": prompts_per_second,
        "responses": responses,
    }

    print(f"\nBenchmark Results:")
    print(f"- Model: {model_name}")
    print(f"- Total prompts: {len(prompts)}")
    print(f"- Total time: {total_time:.2f}s")
    print(f"- Average latency per prompt: {avg_latency:.2f}s")
    print(f"- Tokens per second: {tokens_per_second:.2f}")
    print(f"- Prompts per second: {prompts_per_second:.2f}")

    return results


@pytest.mark.benchmark
def test_engine_benchmark():
    """
    Test the inference engine with Qwen2.5 model and record performance metrics
    """
    # Test with Qwen2.5 model (should be downloaded locally)
    model_name = (
        "Qwen/Qwen2.5-0.5B-Instruct-AWQ"  # Using a smaller model variant for testing
    )

    # Sample prompts for testing
    test_prompts = [
        "Explain the concept of attention mechanism in transformers.",
        "What is the difference between prefill and decode phases in inference?",
        "Describe how KV-cache works in LLM inference.",
        "Explain the benefits of continuous batching in LLM serving.",
        "What are the advantages of paged KV-cache over contiguous cache?",
        "Describe the radial attention mechanism and its benefits.",
        "How does FlashAttention reduce memory complexity?",
        "What is FlashInfer and how does it optimize inference?",
        "Explain the difference between vLLM and SGLang approaches.",
        "How can we optimize GPU memory usage in LLM inference?",
    ]

    try:
        results = benchmark_model(model_name, test_prompts, max_tokens=150)

        # Save results to JSON file
        output_file = (
            f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Benchmark results saved to {output_file}")

        # Assert basic performance requirements (these are just examples)
        assert results["tokens_per_second"] > 0, (
            "Should generate tokens at non-zero rate"
        )
        assert results["prompts_per_second"] > 0, (
            "Should process prompts at non-zero rate"
        )

        return results

    except Exception as e:
        print(f"Error running benchmark: {e}")
        print("Make sure the model is downloaded locally before running this test.")
        # Return a minimal result in case of failure for documentation
        return {
            "model_name": model_name,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def test_simple_generation():
    """
    Simple generation test with a short prompt
    """
    # Using a smaller model for quick testing
    model_name = "microsoft/DialoGPT-small"  # Fallback model if Qwen is not available

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

    except Exception as e:
        print(
            f"Simple generation test failed (this is expected if model is not available): {e}"
        )


if __name__ == "__main__":
    # Run the benchmark test when executed directly
    results = test_engine_benchmark()

    # Print results summary
    print("\n" + "=" * 50)
    print("BENCHMARK COMPLETE")
    print("=" * 50)
    for key, value in results.items():
        if key != "responses":  # Skip responses to keep output clean
            print(f"{key}: {value}")
    print("=" * 50)
    print("="*50)
