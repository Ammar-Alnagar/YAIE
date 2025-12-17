// Flash Attention CUDA Kernel Implementation
// Implements optimized attention with tiling and shared memory

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#define QK_HEAD_DIM 128 // typical head dimension used in attention
#define BLOCK_M 64     // tile size along sequence dimension
#define BLOCK_N 64     // tile size along sequence dimension
#define BLOCK_D 128    // tile size along head dimension

// Utility functions for numerical stability
template<typename scalar_t>
__device__ __forceinline__ scalar_t max3(scalar_t a, scalar_t b, scalar_t c) {
    return fmaxf(a, fmaxf(b, c));
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t max4(scalar_t a, scalar_t b, scalar_t c, scalar_t d) {
    return fmaxf(max3(a, b, c), d);
}

template<typename scalar_t>
__device__ void flash_attention_kernel_impl(
    const int k_num_heads,
    const int k_head_dim,
    const scalar_t* __restrict__ query_ptr,    // [num_tokens, num_heads, head_dim]
    const scalar_t* __restrict__ key_ptr,      // [num_tokens, num_heads, head_dim]
    const scalar_t* __restrict__ value_ptr,    // [num_tokens, num_heads, head_dim]
    scalar_t* __restrict__ output_ptr,         // [num_tokens, num_heads, head_dim]
    const float scale,
    const int q_seq_len,
    const int kv_seq_len
) {
    // Calculate thread indices
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h_idx = blockIdx.y;
    int token_idx = blockIdx.z;

    if (h_idx >= k_num_heads || q_idx >= q_seq_len) return;

    // Shared memory for tiles
    extern __shared__ scalar_t shared_mem[];
    scalar_t* q_tile = shared_mem;
    scalar_t* k_tile = shared_mem + BLOCK_M * BLOCK_D;
    scalar_t* v_tile = shared_mem + BLOCK_M * BLOCK_D + BLOCK_N * BLOCK_D;

    // Initialize local accumulators for m, d, and output
    scalar_t m_prev = -INFINITY;
    scalar_t d_sum = 0.0;
    scalar_t out_local[BLOCK_D];
    for (int d = 0; d < BLOCK_D; d++) {
        out_local[d] = 0.0;
    }

    // Iterate through KV sequence in tiles
    for (int n_start = 0; n_start < kv_seq_len; n_start += BLOCK_N) {
        // Cooperatively load Q tile
        for (int m = 0; m < BLOCK_M; m++) {
            int q_offset = ((token_idx * q_seq_len + (q_idx + m)) * k_num_heads + h_idx) * k_head_dim;
            for (int d = 0; d < BLOCK_D; d++) {
                if (q_idx + m < q_seq_len && d < k_head_dim) {
                    q_tile[m * BLOCK_D + d] = query_ptr[q_offset + d];
                } else {
                    q_tile[m * BLOCK_D + d] = 0.0;
                }
            }
        }

        __syncthreads();

        // Iterate through N dimension of key/value
        for (int n = 0; n < BLOCK_N && n_start + n < kv_seq_len; n++) {
            // Load K and V tiles - each thread loads multiple elements
            for (int d = threadIdx.x; d < BLOCK_D; d += blockDim.x) {
                if (d < k_head_dim) {
                    int kv_offset = ((token_idx * kv_seq_len + (n_start + n)) * k_num_heads + h_idx) * k_head_dim;
                    k_tile[n * BLOCK_D + d] = key_ptr[kv_offset + d];
                    v_tile[n * BLOCK_D + d] = value_ptr[kv_offset + d];
                }
            }

            __syncthreads();

            // Compute attention scores and update accumulators
            for (int m = 0; m < BLOCK_M && q_idx + m < q_seq_len; m++) {
                scalar_t s = 0.0; // raw attention score
                for (int d = 0; d < k_head_dim; d++) {
                    s += q_tile[m * BLOCK_D + d] * k_tile[n * BLOCK_D + d];
                }
                s *= scale; // scale by sqrt(head_dim)

                // Numerical stability: compute m_prime and d_prime
                scalar_t s_prime = s + m_prev;
                scalar_t m_curr = fmaxf(m_prev, s_prime);
                scalar_t exp_s = __expf(s_prime - m_curr);
                scalar_t exp_m = __expf(m_prev - m_curr);

                // Update d_sum
                scalar_t d_curr = d_sum * exp_m + exp_s;

                // Update outputs
                for (int d = 0; d < k_head_dim; d++) {
                    out_local[d] = (d_curr > 0.0) ?
                        (out_local[d] * d_sum * exp_m + exp_s * v_tile[n * BLOCK_D + d]) / d_curr : 0.0;
                }

                // Update m_prev and d_sum for next iteration
                m_prev = m_curr;
                d_sum = d_curr;
            }

            __syncthreads();
        }
    }

    // Store results
    for (int m = 0; m < BLOCK_M && q_idx + m < q_seq_len; m++) {
        for (int d = 0; d < k_head_dim; d++) {
            int out_offset = ((token_idx * q_seq_len + (q_idx + m)) * k_num_heads + h_idx) * k_head_dim + d;
            output_ptr[out_offset] = out_local[d];
        }
    }
}

void flash_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    float softmax_scale
) {
    const auto k_num_heads = query.size(1);
    const auto k_head_dim = query.size(2);
    const auto q_seq_len = query.size(0);
    const auto kv_seq_len = key.size(0);
    const auto num_tokens = query.size(0);  // assuming batch*seq_len flattened

    auto stream = at::cuda::getCurrentCUDAStream();

    // Define grid and block dimensions
    const dim3 block_dim(32, 32, 1);  // threads per block
    const dim3 grid_dim(
        (q_seq_len + BLOCK_M - 1) / BLOCK_M,  // number of Q blocks
        k_num_heads,                         // number of heads
        1                                    // batch dimension (flattened)
    );

    // Calculate shared memory size
    const int shared_mem_size = (BLOCK_M + 2 * BLOCK_N) * BLOCK_D * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "flash_attention_forward_cuda", ([&] {
        flash_attention_kernel_impl<scalar_t>(
            k_num_heads,
            k_head_dim,
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            softmax_scale,
            q_seq_len,
            kv_seq_len
        );
    }));
}
