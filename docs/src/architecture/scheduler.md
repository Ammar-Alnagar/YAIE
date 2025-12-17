# Scheduler Logic: SGLang-Style Request Management

## Overview

The SGLang-style scheduler (`core/sglang_scheduler.py`) implements advanced request scheduling with prefix grouping and computation sharing capabilities. Unlike traditional schedulers, this implementation focuses on maximizing computational efficiency by identifying and leveraging shared prefixes among different requests.

## Key Concepts

### Request States

The scheduler manages requests through several states:

- **PENDING**: New requests awaiting initial processing
- **SCHEDULED_PREFILL**: Requests ready for prefill phase
- **RUNNING_PREFILL**: Currently processing full prompts
- **SCHEDULED_DECODE**: Requests ready for token generation
- **RUNNING_DECODE**: Currently generating tokens
- **COMPLETED**: Finished requests
- **CANCELLED**: Cancelled requests

### Prefix-Based Grouping

The scheduler uses prefix hashing to group requests with common prefixes:

```python
def _calculate_prefix_hash(self, prompt: str) -> Optional[str]:
    # Calculate hash to identify common prefixes
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
```

## Architecture

### Core Data Structures

#### Request Management
```python
# Separate queues for different processing phases
self.pending_requests: List[Request] = []
self.prefill_requests: List[Request] = []
self.running_prefill: List[Request] = []
self.decode_requests: List[Request] = []
self.running_decode: List[Request] = []
self.completed_requests: List[Request] = []
```

#### Prefix Grouping
```python
# Group requests by common prefixes for shared computation
self.prefix_groups: Dict[str, List[Request]] = defaultdict(list)
self.request_lookup: Dict[str, Request] = {}
```

### Scheduling Strategy

The scheduler implements a SGLang-inspired strategy:

1. **Prioritize Decode Requests**: Minimize token-to-token latency
2. **Maximize Prefill Efficiency**: Process new requests efficiently
3. **Leverage Prefix Sharing**: Share computation for similar requests
4. **Memory-Aware Scheduling**: Respect KV-cache limitations

## Detailed Implementation

### Request Lifecycle

#### 1. Request Addition
```python
def add_request(self, prompt: str, max_tokens: int = 128, ...) -> str:
    # Calculate prefix hash for grouping
    prefix_hash = self._calculate_prefix_hash(prompt)
    # Add to prefix group if applicable
    if prefix_hash:
        self.prefix_groups[prefix_hash].append(request)
        request.request_group = prefix_hash
```

#### 2. Scheduling Step
```python
def schedule_step(self) -> Tuple[List[Request], List[Request]]:
    # First, prioritize decode requests
    decode_batch = []
    prefill_batch = []
    
    # Calculate remaining capacity after decode allocation
    remaining_capacity = self.max_batch_size - len(decode_batch)
    
    # Fill remaining capacity with prefill requests
    if remaining_capacity > 0:
        num_prefills = min(len(prefill_candidates), remaining_capacity, self.max_prefill_batch_size)
        prefill_batch = prefill_candidates[:num_prefills]
```

### Batch Selection Policy

The scheduler implements a multi-level priority system:

1. **Decode Priority**: Continue existing generation to minimize latency
2. **Prefill Efficiency**: Process new requests in efficient batches
3. **Memory Management**: Ensure sufficient KV-cache for all requests
4. **Prefix Sharing**: Group similar requests for computation sharing

### Prefill Processing

Prefill requests undergo full prompt processing:

```python
def process_prefill_batch(self, requests: List[Request]) -> List[Request]:
    for req in requests:
        # Process full prompt in one forward pass
        req.status = RequestStatus.SCHEDULED_DECODE
        # Initialize output sequence
        if req.output_ids is None:
            req.output_ids = []
```

### Decode Processing

Decode requests generate tokens one-by-one:

```python
def process_decode_batch(self, requests: List[Request]) -> List[Request]:
    for req in requests:
        # Get logits from model (simplified)
        dummy_logits = torch.randn(1, 1000)
        
        # Sample next token using kernel
        next_token_tensor = self.sampling_kernel.sample(
            dummy_logits,
            temperature=req.temperature,
            top_p=req.top_p
        )
        
        # Update position and check termination
        req.current_position += 1
        if req.current_position >= req.max_tokens:
            req.status = RequestStatus.COMPLETED
```

## SGLang-Style Optimizations

### 1. Computation Sharing

The scheduler identifies requests with shared prefixes:

```python
def find_shared_prefixes(self, token_ids: List[int]) -> Tuple[List[str], List[int]]:
    # Traverse radix tree to find matching prefixes
    # Return requests that can share computation
```

### 2. Memory-Aware Scheduling

The scheduler connects to the memory manager for KV-cache coordination:

```python
def connect_memory_manager(self, memory_manager):
    self.memory_manager = memory_manager
```

### 3. Continuous Batching

The scheduler maintains continuous processing by balancing prefill and decode requests:

- Decode requests have higher priority (latency-sensitive)
- Prefill requests fill remaining batch capacity
- Memory requirements are considered during scheduling

## Performance Considerations

### Batch Size Optimization

The scheduler uses different batch size limits:

- `max_prefill_batch_size`: Limits prefill batch size for memory efficiency
- `max_decode_batch_size`: Larger limit for decode due to smaller memory footprint
- `max_batch_size`: Overall system limit

### Memory Management

The scheduler coordinates with the KV-cache manager to:

- Allocate blocks for new requests
- Track memory usage during processing
- Ensure sufficient memory for scheduled requests

## Integration with Other Components

### Memory Manager Integration

```python
def process_prefill_batch(self, requests: List[Request]) -> List[Request]:
    if self.memory_manager:
        # Allocate KV cache blocks for requests
        pass
```

### Sampling Kernel Integration

```python
def process_decode_batch(self, requests: List[Request]) -> List[Request]:
    # Use sampling kernel for token selection
    next_token_tensor = self.sampling_kernel.sample(...)
```

## Request Status Monitoring

### Queue Status

The scheduler provides detailed status information:

```python
def get_queue_status(self) -> Dict[str, int]:
    return {
        "pending": len(self.pending_requests),
        "prefill_queue": len(self.prefill_requests),
        "running_prefill": len(self.running_prefill),
        "decode_queue": len(self.decode_requests),
        "running_decode": len(self.running_decode),
        "completed": len(self.completed_requests),
        "total_active": self.get_active_request_count(),
    }
```

### Request Result Access

```python
def get_request_result(self, req_id: str) -> Optional[Dict[str, Any]]:
    # Check completed requests for results
```

## Implementation Challenges

### 1. Prefix Hashing

For educational purposes, the implementation uses simple string hashing. In production:

- Use token ID sequences for more accurate prefix matching
- Implement more sophisticated similarity measures
- Consider semantic similarity for better grouping

### 2. Memory Allocation

The current implementation shows integration points for memory management. A full implementation would:

- Calculate precise memory requirements
- Implement cache eviction policies
- Handle memory fragmentation

### 3. Computation Sharing

The radix tree integration points exist but require full implementation of:

- Efficient tree traversal
- Shared computation tracking
- Result distribution to multiple requests

## Scheduling Algorithm Details

### Step-by-Step Process

1. **Decode Prioritization**: Schedule as many decode requests as possible
2. **Capacity Calculation**: Determine remaining batch capacity
3. **Prefill Scheduling**: Fill remaining capacity with prefill requests
4. **Memory Verification**: Confirm sufficient KV-cache availability
5. **Batch Execution**: Process scheduled requests

### Optimization Strategies

The scheduler implements several optimization strategies:

1. **Temporal Multiplexing**: Interleave prefill and decode for efficiency
2. **Spatial Multiplexing**: Group similar requests for shared computation
3. **Memory Multiplexing**: Optimize KV-cache usage across requests

## Future Extensions

The scheduler design supports:

- Advanced prefix matching algorithms
- Dynamic batch size adjustment
- Request preemption and rescheduling
- Multi-GPU coordination
- Custom scheduling policies