// CPU fallback for Flash Attention
#include <torch/extension.h>

torch::Tensor flash_attention_forward_cpu(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    float softmax_scale
) {
    // CPU implementation - direct PyTorch operations
    auto attn_weights = torch::matmul(query, key.transpose(-1, -2)) * softmax_scale;
    attn_weights = torch::softmax(attn_weights, -1);
    return torch::matmul(attn_weights, value);
}

void flash_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    float softmax_scale
) {
    // If CUDA is available, the CUDA implementation will be used
    // Otherwise, use CPU implementation
    if (query.is_cuda()) {
        // This function will be replaced by the CUDA implementation
        // For CPU fallback when CUDA is requested but not available
        auto result = flash_attention_forward_cpu(query, key, value, softmax_scale);
        output.copy_(result);
    } else {
        auto result = flash_attention_forward_cpu(query, key, value, softmax_scale);
        output.copy_(result);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention_forward", &flash_attention_forward, "Flash Attention forward (CUDA)");
}