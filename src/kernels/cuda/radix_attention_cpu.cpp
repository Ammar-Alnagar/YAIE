// Radix Attention CPU Fallback Placeholder
#include <torch/extension.h>

void radix_attention_kernel_cpu(
    torch::Tensor output,
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor radix_indices, 
    int block_size
) {
    // CPU fallback
}
