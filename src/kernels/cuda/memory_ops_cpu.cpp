// Memory Ops CPU Fallback Placeholder
#include <torch/extension.h>

void copy_blocks_kernel_cpu(
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_mapping, 
    int num_mappings
) {
    // CPU fallback
}
