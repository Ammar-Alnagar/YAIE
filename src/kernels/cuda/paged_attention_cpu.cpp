// Paged Attention CPU Fallback Placeholder
#include <torch/extension.h>

void paged_attention_kernel_cpu(
    torch::Tensor output,
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    int block_size,
    int max_context_len
) {
    // CPU fallback
}
