#!/usr/bin/env python3
"""
Integration test for Mini-YAIE kernels with Qwen/Qwen2.5-0.5B-Instruct model
"""

import torch
import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_kv_cache():
    """Test the KV cache functionality"""
    print("Testing KV Cache functionality...")
    from src.kernels.kv_cache import KVCacheManager, KVCacheBlock
    
    # Test parameters
    num_blocks = 10
    block_size = 16
    num_heads = 8
    head_dim = 64
    dtype = torch.float16
    
    # Create KV cache manager
    cache_manager = KVCacheManager(num_blocks, block_size, num_heads, head_dim, dtype)
    
    # Allocate blocks for a request
    request_id = "test_request"
    num_tokens = 25  # Need 2 blocks (16 + 9)
    allocated_blocks = cache_manager.allocate_blocks(request_id, num_tokens)
    
    print(f"Allocated blocks: {allocated_blocks}")
    assert len(allocated_blocks) == 2, "Should allocate 2 blocks for 25 tokens"
    
    # Test copying blocks
    src_blocks = allocated_blocks
    dst_blocks = cache_manager.allocate_blocks("dst_request", num_tokens)
    cache_manager.copy_blocks(src_blocks, dst_blocks)

    # Test freeing blocks
    cache_manager.free_blocks(request_id)
    cache_manager.free_blocks("dst_request")
    
    print("✓ KV Cache functionality test passed")


def test_radix_tree():
    """Test the radix tree functionality for prefix matching"""
    print("Testing Radix Tree functionality...")
    from src.kernels.radix_tree import RadixTree, RequestPrefixMatcher
    
    # Create a radix tree
    radix_tree = RadixTree()
    
    # Add some test requests with overlapping prefixes
    request1_tokens = [1, 2, 3, 4, 5]  # "Hello world"
    request2_tokens = [1, 2, 3, 6, 7]  # "Hello there"
    request3_tokens = [8, 9, 10]       # "Goodbye"
    
    radix_tree.insert_request("req1", request1_tokens)
    radix_tree.insert_request("req2", request2_tokens)
    radix_tree.insert_request("req3", request3_tokens)
    
    # Test prefix matching
    new_tokens = [1, 2, 3, 11, 12]
    shared_requests, shared_length = radix_tree.find_shared_prefixes(new_tokens)
    
    print(f"Shared requests: {shared_requests}, shared length: {shared_length}")
    assert "req1" in shared_requests and "req2" in shared_requests, "Should find req1 and req2 as shared"
    assert shared_length == 3, "Should have 3 tokens of shared prefix"
    
    # Create prefix matcher
    matcher = RequestPrefixMatcher()
    matcher.add_request_prefix("req4", [1, 2, 3, 4, 5, 6])
    matcher.add_request_prefix("req5", [1, 2, 3, 7, 8, 9])
    
    sharing_opportunities = matcher.find_computation_sharing_opportunities("new_req", [1, 2, 3, 10])
    print(f"Sharing opportunities: {sharing_opportunities}")
    
    # Get optimization suggestions
    suggestions = matcher.get_optimization_suggestions()
    print(f"Sharing graph has {len(suggestions['sharing_graph'])} sharing opportunities")
    
    print("✓ Radix Tree functionality test passed")


def test_sampling():
    """Test the sampling functionality"""
    print("Testing Sampling functionality...")
    from src.kernels.sampling import SamplingKernel
    
    # Create a sampling kernel
    sampler = SamplingKernel()
    
    # Create test logits (batch_size=2, vocab_size=1000)
    torch.manual_seed(42)
    logits = torch.randn(2, 1000)
    
    # Test basic sampling
    sampled_ids = sampler.sample(logits)
    assert sampled_ids.shape == (2,), f"Expected shape (2,), got {sampled_ids.shape}"
    
    # Test with temperature
    sampled_ids_temp = sampler.sample(logits, temperature=0.7)
    assert sampled_ids_temp.shape == (2,), f"Expected shape (2,), got {sampled_ids_temp.shape}"
    
    # Test with top-p
    sampled_ids_topp = sampler.sample(logits, top_p=0.9)
    assert sampled_ids_topp.shape == (2,), f"Expected shape (2,), got {sampled_ids_topp.shape}"
    
    # Test with top-k
    sampled_ids_topk = sampler.sample(logits, top_k=50)
    assert sampled_ids_topk.shape == (2,), f"Expected shape (2,), got {sampled_ids_topk.shape}"
    
    # Test with all parameters
    sampled_ids_all = sampler.sample(logits, temperature=0.8, top_p=0.9, top_k=100)
    assert sampled_ids_all.shape == (2,), f"Expected shape (2,), got {sampled_ids_all.shape}"
    
    print("✓ Sampling functionality test passed")


def test_model_integration():
    """Test integration with Qwen2.5-0.5B model"""
    print("Testing model integration...")
    
    # Download the model if not present
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    # Check if model is already downloaded
    model_path = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
    if not os.path.exists(model_path):
        print("Model not found in cache, downloading...")
        snapshot_download(repo_id=model_name, cache_dir=cache_dir)
    else:
        print("Model found in cache, proceeding...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Create a simple test prompt
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move to GPU if available
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate text with the original model
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")
    
    # Test that our kernels can work with the model's dimensions
    config = model.config
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    head_dim = hidden_size // num_attention_heads  # Should be 64 for Qwen2.5-0.5B
    max_position_embeddings = config.max_position_embeddings
    
    print(f"Model config: hidden_size={hidden_size}, num_heads={num_attention_heads}, head_dim={head_dim}")
    
    # Test RadixAttentionBlock with model parameters
    from src.kernels.radix_attention import RadixAttentionBlock
    
    radix_attn = RadixAttentionBlock(
        hidden_size=hidden_size,
        num_heads=num_attention_heads,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings
    ).to(device)
    
    # Create test inputs matching model dimensions
    batch_size = 1
    seq_len = inputs['input_ids'].size(1)  # Length of input prompt
    test_hidden = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float16)
    
    # Test forward pass through radial attention
    attn_output, attn_weights, past_key_value = radix_attn(
        hidden_states=test_hidden,
        attention_mask=torch.ones(batch_size, seq_len, device=device),
        use_cache=True
    )
    
    print(f"Radix attention output shape: {attn_output.shape}")
    assert attn_output.shape == test_hidden.shape, "Output shape should match input shape"
    
    print("✓ Model integration test passed")


def test_paged_attention_concept():
    """Test the paged attention concept with our implemented cache"""
    print("Testing Paged Attention concept...")

    # Test the cache manager that will be used in paged attention
    from src.kernels.kv_cache import KVCacheManager
    
    # Create a cache manager similar to what would be used in paged attention
    cache_manager = KVCacheManager(
        num_blocks=20,
        block_size=16,
        num_heads=8,
        head_dim=64,
        dtype=torch.float16
    )
    
    # Simulate multiple requests
    req1_tokens = 25  # Will need 2 blocks
    req2_tokens = 40  # Will need 3 blocks
    req3_tokens = 10  # Will need 1 block
    
    blocks_req1 = cache_manager.allocate_blocks("req1", req1_tokens)
    blocks_req2 = cache_manager.allocate_blocks("req2", req2_tokens)
    blocks_req3 = cache_manager.allocate_blocks("req3", req3_tokens)
    
    print(f"Request 1 blocks: {blocks_req1} (for {req1_tokens} tokens)")
    print(f"Request 2 blocks: {blocks_req2} (for {req2_tokens} tokens)")
    print(f"Request 3 blocks: {blocks_req3} (for {req3_tokens} tokens)")
    
    # Verify that all allocations worked
    total_used = len(blocks_req1) + len(blocks_req2) + len(blocks_req3)
    print(f"Total blocks used: {total_used}")
    
    # Test the get_kv_tensors method
    all_blocks = blocks_req1 + blocks_req2 + blocks_req3
    seq_lens = [req1_tokens, req2_tokens, req3_tokens]
    
    # This should retrieve the tensors from the allocated blocks
    keys, values = cache_manager.get_kv_tensors(all_blocks, seq_lens)
    print(f"Retrieved keys shape: {keys.shape}")
    print(f"Retrieved values shape: {values.shape}")
    
    # Free resources
    cache_manager.free_blocks("req1")
    cache_manager.free_blocks("req2")
    cache_manager.free_blocks("req3")
    
    print("✓ Paged Attention concept test passed")


def run_all_tests():
    """Run all integration tests"""
    print("Running Mini-YAIE kernel integration tests...\n")
    
    try:
        test_kv_cache()
        print()
        
        test_radix_tree()
        print()
        
        test_sampling()
        print()
        
        test_paged_attention_concept()
        print()
        
        test_model_integration()
        print()
        
        print("🎉 All integration tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        exit(1)