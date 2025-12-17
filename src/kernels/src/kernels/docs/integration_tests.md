# Integration Tests (`test_kernels_integration.py`)

The `test_kernels_integration.py` module contains integration tests for the core components of the Mini-YAIE kernels. These tests are designed to verify that individual kernel modules (`kv_cache`, `radix_tree`, `sampling`, `radix_attention`) function correctly in isolation and, crucially, that they integrate properly with external libraries like HuggingFace Transformers and with each other. The tests use the `Qwen/Qwen2.5-0.5B-Instruct` model as a concrete example for integration.

## How to Run Tests

To execute all integration tests, simply run the Python script from the command line:

```bash
python3 test_kernels_integration.py
```

The script will print progress and results for each test function. If all tests pass, it will output "All integration tests passed successfully!". In case of failure, it will print an error message and a traceback.

## Test Functions

### `test_kv_cache()`

*   **Purpose**: Verifies the basic functionality of the `KVCacheManager` and `KVCacheBlock` from the `src.kernels.kv_cache` module.
*   **Key Aspects Tested**:
    *   **Initialization**: Proper instantiation of `KVCacheManager`.
    *   **Block Allocation**: Correct allocation of `KVCacheBlock`s for a given number of tokens. It checks that the expected number of blocks are allocated.
    *   **Block Copying**: Tests the `copy_blocks` method to ensure KV data can be successfully transferred between allocated blocks.
    *   **Block Freeing**: Verifies that `free_blocks` correctly returns blocks to the free list.

### `test_radix_tree()`

*   **Purpose**: Validates the `RadixTree` and `RequestPrefixMatcher` components from `src.kernels.radix_tree`, focusing on prefix matching and sharing logic.
*   **Key Aspects Tested**:
    *   **Request Insertion**: Ability to insert token sequences for multiple requests into the radix tree.
    *   **Prefix Matching**: Correct identification of requests that share common prefixes with a new token sequence, along with the length of the shared prefix.
    *   **Computation Sharing Opportunities**: Checks the `RequestPrefixMatcher`'s ability to find computation sharing opportunities.
    *   **Optimization Suggestions**: Ensures `get_optimization_suggestions` provides relevant information.

### `test_sampling()`

*   **Purpose**: Confirms the correct operation of the `SamplingKernel` from `src.kernels.sampling`, testing various token sampling strategies.
*   **Key Aspects Tested**:
    *   **Basic Sampling**: Verifies that the sampler can draw token IDs from raw logits.
    *   **Temperature Scaling**: Tests sampling behavior when a `temperature` parameter is applied.
    *   **Top-P (Nucleus) Sampling**: Checks filtering based on cumulative probability `top_p`.
    *   **Top-K Sampling**: Verifies filtering based on the `top_k` most probable tokens.
    *   **Combined Parameters**: Ensures all sampling parameters (`temperature`, `top_p`, `top_k`) can be used simultaneously.

### `test_model_integration()`

*   **Purpose**: An end-to-end test verifying the seamless integration of `RadixAttentionBlock` with a pre-trained HuggingFace model (`Qwen/Qwen2.5-0.5B-Instruct`).
*   **Key Aspects Tested**:
    *   **Model Download & Loading**: Ensures the specified model and tokenizer can be downloaded and loaded correctly.
    *   **Text Generation**: Performs a basic text generation task using the loaded model.
    *   **`RadixAttentionBlock` Integration**:
        *   Initializes `RadixAttentionBlock` with parameters derived from the loaded model's configuration (`hidden_size`, `num_attention_heads`, `head_dim`).
        *   Performs a forward pass through the `RadixAttentionBlock` using dummy hidden states with correct dimensions, verifying output shapes and the ability to cache key-value pairs.

### `test_paged_attention_concept()`

*   **Purpose**: Validates the conceptual basis of paged attention using the `KVCacheManager`. This test focuses on the memory management aspects that underpin paged attention.
*   **Key Aspects Tested**:
    *   **Cache Manager Initialization**: Proper setup of `KVCacheManager`.
    *   **Multi-Request Block Allocation**: Allocates KV-cache blocks for several simulated requests, each requiring a different number of blocks.
    *   **KV Tensor Retrieval**: Verifies that `get_kv_tensors` can correctly retrieve and concatenate KV data from multiple allocated blocks for different sequences.
    *   **Resource Freeing**: Ensures `free_blocks` can deallocate blocks for individual requests.

## `run_all_tests()`

This function orchestrates the execution of all the individual test functions defined in the module.

*   **Purpose**: Provides a single entry point to run the entire suite of integration tests.
*   **Error Handling**: Includes a `try-except` block to catch any exceptions during test execution, print an error message, and provide a traceback for debugging.

## Main Execution Block

The `if __name__ == "__main__":` block ensures that `run_all_tests()` is called when the script is executed directly, allowing for easy command-line testing. If any test fails, the script exits with a non-zero status code (`exit(1)`), which is useful for CI/CD pipelines.