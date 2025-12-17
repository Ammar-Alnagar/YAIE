// CPU fallback for Paged Attention
#include <torch/extension.h>

torch::Tensor paged_attention_cpu(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    int block_size,
    int max_context_len
) {
    // CPU implementation of paged attention
    auto num_seqs = query.size(0);
    auto num_heads = query.size(1);
    auto head_dim = query.size(2);
    
    auto output = torch::zeros_like(query);
    
    // Simple CPU implementation
    for (int seq_idx = 0; seq_idx < num_seqs; seq_idx++) {
        auto context_len = context_lens[seq_idx].item<int>();
        auto seq_query = query[seq_idx];  // [num_heads, head_dim]
        
        torch::Tensor seq_attn_weights = torch::zeros({num_heads, context_len});
        
        // For each position in the context
        for (int pos = 0; pos < context_len; pos++) {
            auto block_idx = pos / block_size;
            auto pos_in_block = pos % block_size;
            auto physical_block_idx = block_tables[seq_idx][block_idx].item<int>();
            
            // Get keys for this position
            auto pos_keys = key_cache[physical_block_idx][pos_in_block];  // [num_heads, head_dim]
            
            // Calculate attention scores
            auto scores = (seq_query * pos_keys).sum(-1);  // [num_heads]
            seq_attn_weights.index_put_({torch::indexing::Slice(), pos}, scores);
        }
        
        // Apply softmax
        seq_attn_weights = torch::softmax(seq_attn_weights, -1);  // [num_heads, context_len]
        
        // Calculate output
        torch::Tensor seq_output = torch::zeros({num_heads, head_dim});
        for (int pos = 0; pos < context_len; pos++) {
            auto block_idx = pos / block_size;
            auto pos_in_block = pos % block_size;
            auto physical_block_idx = block_tables[seq_idx][block_idx].item<int>();
            
            // Get values for this position
            auto pos_values = value_cache[physical_block_idx][pos_in_block];  // [num_heads, head_dim]
            auto attn_scores = seq_attn_weights.index({torch::indexing::Slice(), pos}).unsqueeze(-1);  // [num_heads, 1]
            
            seq_output += attn_scores * pos_values;  // [num_heads, head_dim]
        }
        
        output[seq_idx] = seq_output;
    }
    
    return output;
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
    if (query.is_cuda()) {
        // This function will be replaced by the CUDA implementation
        auto result = paged_attention_cpu(query, key_cache, value_cache, block_tables, context_lens, block_size, max_context_len);
        output.copy_(result);
    } else {
        auto result = paged_attention_cpu(query, key_cache, value_cache, block_tables, context_lens, block_size, max_context_len);
        output.copy_(result);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_attention", &paged_attention_kernel, "Paged Attention (CUDA)");
}