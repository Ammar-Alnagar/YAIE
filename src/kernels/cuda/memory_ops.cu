// Memory Operations CUDA Kernel Implementation
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#define BLOCK_SIZE 16
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

__global__ void copy_blocks_kernel_impl(
    const float* __restrict__ src_key_ptr,
    const float* __restrict__ src_value_ptr,
    float* __restrict__ dst_key_ptr,
    float* __restrict__ dst_value_ptr,
    const int* __restrict__ block_mapping_ptr,  // [num_mappings, 2] - [src_block_id, dst_block_id]
    const int num_blocks,
    const int block_size,
    const int num_heads,
    const int head_dim
) {
    // Get thread indices
    const int mapping_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;

    if (mapping_idx >= num_blocks) return;

    // Get source and destination block IDs
    const int src_block_id = block_mapping_ptr[mapping_idx * 2];
    const int dst_block_id = block_mapping_ptr[mapping_idx * 2 + 1];

    // Calculate total elements per block
    const int elements_per_block = block_size * num_heads * head_dim;
    const int total_threads = blockDim.x;

    // Each thread copies multiple elements
    for (int elem_idx = thread_idx; elem_idx < elements_per_block; elem_idx += total_threads) {
        // Calculate offsets for source and destination
        const int src_offset = src_block_id * elements_per_block + elem_idx;
        const int dst_offset = dst_block_id * elements_per_block + elem_idx;

        // Copy key values
        dst_key_ptr[dst_offset] = src_key_ptr[src_offset];

        // Copy value values
        dst_value_ptr[dst_offset] = src_value_ptr[src_offset];
    }
}

__global__ void copy_single_block_kernel_impl(
    const float* __restrict__ src_key_ptr,
    const float* __restrict__ src_value_ptr,
    float* __restrict__ dst_key_ptr,
    float* __restrict__ dst_value_ptr,
    const int src_block_id,
    const int dst_block_id,
    const int block_size,
    const int num_heads,
    const int head_dim
) {
    // Calculate total elements to copy
    const int elements_per_block = block_size * num_heads * head_dim;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < elements_per_block) {
        // Calculate offsets
        const int src_offset = src_block_id * elements_per_block + tid;
        const int dst_offset = dst_block_id * elements_per_block + tid;

        // Copy key and value
        dst_key_ptr[dst_offset] = src_key_ptr[src_offset];
        dst_value_ptr[dst_offset] = src_value_ptr[src_offset];
    }
}

__global__ void swap_blocks_kernel_impl(
    float* __restrict__ key_cache_ptr,
    float* __restrict__ value_cache_ptr,
    const int* __restrict__ block_mapping_ptr,  // [num_mappings, 2] - [block_id_1, block_id_2] for swapping
    const int num_swaps,
    const int block_size,
    const int num_heads,
    const int head_dim
) {
    // Get thread indices
    const int swap_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;

    if (swap_idx >= num_swaps) return;

    // Get block IDs to swap
    const int block_id_1 = block_mapping_ptr[swap_idx * 2];
    const int block_id_2 = block_mapping_ptr[swap_idx * 2 + 1];

    // Calculate total elements per block
    const int elements_per_block = block_size * num_heads * head_dim;
    const int total_threads = blockDim.x;

    // Each thread swaps multiple elements
    for (int elem_idx = thread_idx; elem_idx < elements_per_block; elem_idx += total_threads) {
        // Calculate offsets for both blocks
        const int offset_1 = block_id_1 * elements_per_block + elem_idx;
        const int offset_2 = block_id_2 * elements_per_block + elem_idx;

        // Swap key values
        float temp_key = key_cache_ptr[offset_1];
        key_cache_ptr[offset_1] = key_cache_ptr[offset_2];
        key_cache_ptr[offset_2] = temp_key;

        // Swap value values
        float temp_val = value_cache_ptr[offset_1];
        value_cache_ptr[offset_1] = value_cache_ptr[offset_2];
        value_cache_ptr[offset_2] = temp_val;
    }
}

void copy_blocks_kernel(
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_mapping,
    int num_mappings
) {
    // Extract tensor dimensions
    const auto block_size = key_cache.size(1);
    const auto num_heads = key_cache.size(2);
    const auto head_dim = key_cache.size(3);

    // Calculate grid and block dimensions
    dim3 grid(num_mappings, 1, 1);
    dim3 block(min(MAX_THREADS_PER_BLOCK, 256), 1, 1);  // Use 256 threads per block

    auto stream = at::cuda::getCurrentCUDAStream();

    // Call the kernel implementation
    copy_blocks_kernel_impl<<<grid, block, 0, stream>>>(
        key_cache.data_ptr<float>(),       // src_key_ptr
        value_cache.data_ptr<float>(),     // src_value_ptr
        key_cache.data_ptr<float>(),       // dst_key_ptr (we assume this points to destination)
        value_cache.data_ptr<float>(),     // dst_value_ptr (we assume this points to destination)
        block_mapping.data_ptr<int>(),     // block_mapping_ptr
        num_mappings,                      // num_blocks
        block_size,                        // block_size
        num_heads,                         // num_heads
        head_dim                           // head_dim
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error in copy_blocks_kernel: %s\n", cudaGetErrorString(err));
    }
}

// Additional function for swapping blocks (useful for cache management)
void swap_blocks_kernel(
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_mapping,
    int num_swaps
) {
    // Extract tensor dimensions
    const auto block_size = key_cache.size(1);
    const auto num_heads = key_cache.size(2);
    const auto head_dim = key_cache.size(3);

    // Calculate grid and block dimensions
    dim3 grid(num_swaps, 1, 1);
    dim3 block(min(MAX_THREADS_PER_BLOCK, 256), 1, 1);

    auto stream = at::cuda::getCurrentCUDAStream();

    // Call the kernel implementation
    swap_blocks_kernel_impl<<<grid, block, 0, stream>>>(
        key_cache.data_ptr<float>(),
        value_cache.data_ptr<float>(),
        block_mapping.data_ptr<int>(),
        num_swaps,
        block_size,
        num_heads,
        head_dim
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error in swap_blocks_kernel: %s\n", cudaGetErrorString(err));
    }
}
