# Engine Orchestration

## Overview

The Inference Engine serves as the main orchestrator of the Mini-YAIE system, coordinating between various components to provide a unified interface for LLM inference. The engine implements SGLang-style continuous batching with radial attention and prefix sharing to maximize efficiency and throughput.

## Engine Architecture

The main engine is implemented in `src/engine.py` and follows a modular design pattern where each component is responsible for specific aspects of request processing:

```
┌─────────────────┐
│   API Layer     │  ← Requests enter here
├─────────────────┤
│ Engine Orchestration │  ← Coordination happens here
├─────────────────┤
│   Scheduler     │  ← Request scheduling
├─────────────────┤
│  Memory Manager │  ← KV-cache management
├─────────────────┤
│  Attention Core │  ← Radial attention computation
├─────────────────┤
│  Model/Kernel   │  ← Forward pass execution
└─────────────────┘
```

## Core Components

### 1. Model Loading Integration

The engine handles model and tokenizer loading through the ModelLoader component:

```python
def __init__(self, model_name: str):
    self.tokenizer: PreTrainedTokenizer = self._load_tokenizer()
    self.model = self._load_model()
```

This ensures that models are properly loaded from HuggingFace or local cache with appropriate configuration.

### 2. SGLang-Style Scheduler

The engine integrates with the SGLangScheduler for advanced request scheduling:

```python
self.scheduler = SGLangScheduler(
    max_batch_size=8, 
    max_prefill_batch_size=16, 
    max_decode_batch_size=256
)
```

The scheduler implements prefix grouping and multi-step processing for computation sharing.

### 3. Radial Attention System

The engine connects to the radial attention mechanism:

```python
self.radix_attention = RadixAttentionWithPagedKVCache(
    num_layers=self.model.config.num_hidden_layers,
    num_heads=self.model.config.num_attention_heads,
    head_dim=self.model.config.hidden_size // self.model.config.num_attention_heads,
)
```

### 4. KV-Cache Management

The engine manages memory through the KVCacheManager:

```python
self.kv_cache_manager = KVCacheManager(
    num_blocks=2000,
    block_size=16,
    num_heads=self.model.config.num_attention_heads,
    head_dim=self.model.config.hidden_size // self.model.config.num_attention_heads,
    dtype=torch.float16,
)
```

## Request Processing Flow

### 1. Request Addition

```python
def generate(self, prompts: List[str], **kwargs) -> List[str]:
    # Add requests to scheduler
    request_ids = []
    for prompt in prompts:
        req_id = self.scheduler.add_request(prompt, **kwargs)
        request_ids.append(req_id)
```

### 2. Generation Loop

The engine runs a main generation loop that processes requests:

```python
def _run_generation_loop(self, request_ids: List[str]) -> List[str]:
    # Process requests in batches
    # Handle prefill and decode phases
    # Manage KV-cache efficiently
```

### 3. Response Generation

The engine generates responses with proper tokenization and formatting:

```python
# Generate response using the existing generate method
responses = self.generate([formatted_prompt], **kwargs)
generated_text = responses[0] if responses else ""
```

## SGLang-Style Optimization Features

### 1. Continuous Batching

The engine supports continuous batching where requests at different stages can be processed together:

- Prefill requests (processing full prompts)
- Decode requests (generating single tokens)
- Mixed batches combining both types

### 2. Prefix Sharing

The engine enables computation sharing for requests with common prefixes:

- Radix tree identifies shared prefixes
- Common computations are performed once
- Results are shared among multiple requests

### 3. Memory Efficiency

The engine optimizes memory usage through:

- Paged KV-cache management
- Block allocation strategies
- Memory reclamation for completed requests

## API Integration

### 1. Chat Completion API

The engine provides OpenAI-compatible chat completion:

```python
def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    # Format messages using chat template
    # Process through generation pipeline
    # Return in OpenAI format
```

### 2. Streaming Support

The engine supports streaming responses for real-time applications:

```python
def chat_completion_stream(self, messages: List[Dict[str, str]], **kwargs):
    # Generate tokens one by one
    # Yield chunks immediately
    # Maintain OpenAI stream format
```

## Performance Optimization

### 1. Batch Size Management

The engine dynamically adjusts batch sizes based on available memory and request characteristics:

- Prefill batches: Optimized for prompt processing
- Decode batches: Optimized for token generation
- Mixed batches: Balanced between both phases

### 2. Memory Management

The engine coordinates memory usage across components:

```python
# Connect scheduler to memory manager for optimization
self.scheduler.connect_memory_manager(self.kv_cache_manager)
```

### 3. Computation Sharing

The engine maximizes computation sharing through radix attention:

- Shared prefix processing
- Common token computations
- Reduced redundant calculations

## Error Handling and Resilience

### 1. Request Validation

The engine validates requests before processing:

- Input format validation
- Parameter range checking
- Resource availability verification

### 2. Graceful Degradation

When resources are constrained, the engine gracefully degrades:

- Reduced batch sizes
- Fallback mechanisms
- Proper error reporting

### 3. Resource Management

The engine manages system resources effectively:

- GPU memory monitoring
- Request queue management
- Memory cleanup for completed requests

## Integration Points

### 1. Model Interface

The engine interfaces with any HuggingFace-compatible model:

```python
outputs = self.model(current_ids)  # Standard HuggingFace model interface
```

### 2. Sampling Integration

The engine uses the sampling kernel for token generation:

```python
sampling_kernel = SamplingKernel()
next_token_id = sampling_kernel.sample(
    next_token_logits,
    temperature=request.temperature,
    top_p=request.top_p
)
```

### 3. Scheduler Integration

The engine coordinates closely with the scheduler:

```python
# Add requests to scheduler
req_id = self.scheduler.add_request(prompt, **kwargs)
# Process in generation loop
responses = self._run_generation_loop(request_ids)
```

## Engine Configuration

The engine supports various configuration options:

- Model selection and loading
- Batch size limits
- Memory allocation settings
- Performance optimization parameters

## Future Extensions

The engine design supports:

- Additional optimization techniques
- New attention mechanisms
- Enhanced scheduling algorithms
- Advanced memory management strategies