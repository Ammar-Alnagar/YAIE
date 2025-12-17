// Paged Attention CUDA Kernel Implementation
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// Block size constants
#define MAX_SEQ_LEN 4096
#define MAX_BLOCK_SIZE 16
#define WARP_SIZE 32

__global__ void paged_attention_kernel_impl(
    const float* __restrict__ query_ptr,           // [num_seqs, num_heads, head_dim]
    const float* __restrict__ key_cache_ptr,       // [num_blocks, block_size, num_heads, head_dim]
    const float* __restrict__ value_cache_ptr,     // [num_blocks, block_size, num_heads, head_dim]
    const int* __restrict__ block_tables_ptr,      // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens_ptr,      // [num_seqs]
    float* __restrict__ output_ptr,                // [num_seqs, num_heads, head_dim]
    float* __restrict__ exp_sums_ptr,              // [num_seqs, max_num_partitions]
    float* __restrict__ max_logits_ptr,            // [num_seqs, max_num_partitions]
    const float scale,
    const int num_seqs,
    const int num_heads,
    const int head_dim,
    const int block_size,
    const int max_num_blocks_per_seq,
    const int max_context_len,
    const int max_num_partitions
) {
    // Get thread indices
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int block_idx = threadIdx.x / WARP_SIZE;
    const int tid = threadIdx.x % WARP_SIZE;

    if (seq_idx >= num_seqs || head_idx >= num_heads) return;

    const int context_len = context_lens_ptr[seq_idx];
    const int num_blocks = (context_len + block_size - 1) / block_size;

    if (block_idx >= num_blocks) return;

    // Initialize local variables for attention computation
    float sum_exp = 0.0f;
    float max_logit = -INFINITY;
    float out_val = 0.0f;

    // Get the physical block ID from the block table
    const int physical_block_idx = block_tables_ptr[seq_idx * max_num_blocks_per_seq + block_idx];

    // Calculate offset in the cache
    const int cache_offset = physical_block_idx * block_size * num_heads * head_dim;

    // Iterate through positions in the current block
    for (int pos = 0; pos < block_size && (block_idx * block_size + pos) < context_len; pos++) {
        // Calculate the position in the sequence
        const int seq_pos = block_idx * block_size + pos;

        // Compute attention score: dot product between query and key
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            const int query_offset = (seq_idx * num_heads + head_idx) * head_dim + d;
            const int key_offset = cache_offset + pos * num_heads * head_dim + head_idx * head_dim + d;

            score += query_ptr[query_offset] * key_cache_ptr[key_offset];
        }
        score *= scale;  // Apply scaling factor

        // Apply causal mask (for autoregressive decoding)
        if (seq_pos >= context_len) {
            score = -INFINITY;
        }

        // Numerical stability: update max logit for softmax
        const float old_max_logit = max_logit;
        max_logit = fmaxf(max_logit, score);

        // Compute scaling factors
        const float exp_score = __expf(score - max_logit);
        const float scale_factor = __expf(old_max_logit - max_logit);

        // Update sum of exponentials
        sum_exp = sum_exp * scale_factor + exp_score;

        // Update output values
        for (int d = tid; d < head_dim; d += WARP_SIZE) {
            const int out_offset = (seq_idx * num_heads + head_idx) * head_dim + d;
            const int value_offset = cache_offset + pos * num_heads * head_dim + head_idx * head_dim + d;

            // Apply scaling to value
            float value = value_cache_ptr[value_offset];
            out_val = out_val * scale_factor + exp_score * value;
        }
    }

    // Store temporary results for reduction
    // In a real implementation, we would use shared memory and block-level reductions
    // For simplicity, we'll do a basic implementation
    __shared__ float shared_sum_exp[32];  // WARP_SIZE elements
    __shared__ float shared_max_logit[32];
    __shared__ float shared_out_val[32];

    shared_sum_exp[tid] = sum_exp;
    shared_max_logit[tid] = max_logit;
    shared_out_val[tid] = out_val;

    __syncthreads();

    // Simple reduction within block
    for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            float other_max = shared_max_logit[tid + stride];
            if (other_max > shared_max_logit[tid]) {
                shared_sum_exp[tid] = shared_sum_exp[tid] * __expf(shared_max_logit[tid] - other_max) + shared_sum_exp[tid + stride];
                shared_max_logit[tid] = other_max;
            } else {
                shared_sum_exp[tid] = shared_sum_exp[tid] + shared_sum_exp[tid + stride] * __expf(other_max - shared_max_logit[tid]);
            }
            shared_out_val[tid] = shared_out_val[tid] + shared_out_val[tid + stride];
        }
        __syncthreads();
    }

    // Write results
    if (tid == 0) {
        const int out_idx = (seq_idx * num_heads + head_idx) * head_dim;
        for (int d = 0; d < head_dim; d++) {
            output_ptr[out_idx + d] = shared_out_val[0] / fmaxf(shared_sum_exp[0], 1e-8f);
        }
    }
}

void paged_attention_kernel(
    torch::Tensor output,
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    int block_size,
    int max_context_len
) {
    const auto num_seqs = query.size(0);
    const auto num_heads = query.size(1);
    const auto head_dim = query.size(2);
    const auto num_blocks = key_cache.size(0);
    const auto max_num_blocks_per_seq = block_tables.size(1);
    const auto max_num_partitions = (max_context_len + 511) / 512;  // 512 is typical partition size

    // Calculate grid and block dimensions
    dim3 grid(num_seqs, num_heads, 1);
    dim3 block(WARP_SIZE * ((max_context_len + block_size - 1) / block_size), 1, 1);

    // Limit block size to avoid exceeding limits
    if (block.x > 1024) {
        block.x = 1024;
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    // Allocate temporary tensors for attention computation
    auto exp_sums = torch::zeros({num_seqs, max_num_partitions}, torch::dtype(torch::kFloat32).device(key_cache.device()));
    auto max_logits = torch::zeros({num_seqs, max_num_partitions}, torch::dtype(torch::kFloat32).device(key_cache.device()));

    // Scaling factor based on head dimension
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    paged_attention_kernel_impl<<<grid, block, 0, stream>>>(
        query.data_ptr<float>(),
        key_cache.data_ptr<float>(),
        value_cache.data_ptr<float>(),
        block_tables.data_ptr<int>(),
        context_lens.data_ptr<int>(),
        output.data_ptr<float>(),
        exp_sums.data_ptr<float>(),
        max_logits.data_ptr<float>(),
        scale,
        num_seqs,
        num_heads,
        head_dim,
        block_size,
        max_num_blocks_per_seq,
        max_context_len,
        max_num_partitions
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
}
