# Memory Management: Paged KV-Cache System

## Overview

The memory management system in Mini-YAIE implements a paged KV-cache mechanism inspired by systems like vLLM and SGLang. This approach addresses the memory fragmentation challenges in LLM inference by using fixed-size memory blocks (pages) that can be allocated and deallocated independently for each request.

## Core Concepts

### Paged Memory Architecture

Traditional KV-cache management allocates contiguous memory blocks for each request, leading to fragmentation when requests have varying lengths. Paged KV-cache solves this by:

- Dividing the KV-cache into fixed-size blocks (pages)
- Allowing requests to use non-contiguous memory blocks
- Enabling efficient memory reuse and sharing

### Key Benefits

1. **Reduced Fragmentation**: Fixed-size blocks prevent memory fragmentation
2. **Efficient Memory Utilization**: Unused blocks can be allocated to other requests
3. **Scalability**: Supports variable-length requests without memory waste
4. **Computation Sharing**: Enables shared prefixes to use the same memory blocks

## Architecture

### KVCacheBlock Class

Each `KVCacheBlock` represents a fixed-size memory block:

```python
class KVCacheBlock:
    def __init__(self, block_id: int, size: int, num_heads: int, head_dim: int, ...):
        self.block_id = block_id  # Unique identifier for the block
        self.size = size          # Number of tokens this block can hold
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.keys = None          # [size, num_heads, head_dim] tensor
        self.values = None        # [size, num_heads, head_dim] tensor
```

### KVCacheManager Class

The main manager orchestrates all memory operations:

```python
class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int, num_heads: int, head_dim: int, ...):
        self.num_blocks = num_blocks      # Total number of blocks in the pool
        self.block_size = block_size      # Size of each block in tokens
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.blocks: List[KVCacheBlock] = []  # Pool of all blocks
        self.free_block_list: List[int] = []  # Available blocks for allocation
        self.block_tables: dict = {}      # Maps request_id to list of block_ids
```

## Memory Management Operations

### 1. Block Allocation

When a request needs KV-cache memory, the manager allocates the required number of blocks:

```python
def allocate_blocks(self, request_id: str, num_tokens: int) -> List[int]:
    # Calculate required blocks
    num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
    
    # Check availability
    if len(self.free_block_list) < num_blocks_needed:
        raise RuntimeError("Not enough free blocks")
    
    # Allocate from free list
    allocated_block_ids = []
    for _ in range(num_blocks_needed):
        block_id = self.free_block_list.pop(0)  # Remove from free list
        allocated_block_ids.append(block_id)
        self.blocks[block_id].allocate()  # Allocate GPU memory
    
    # Track allocation
    self.block_tables[request_id] = allocated_block_ids
    return allocated_block_ids
```

**Key Aspects:**
- Calculates minimum blocks needed based on token count and block size
- Ensures sufficient free blocks before allocation
- Updates free block list and request tracking
- Actually allocates GPU memory for the blocks

### 2. Block Deallocation

When requests complete, their blocks are returned to the free pool:

```python
def free_blocks(self, request_id: str):
    if request_id in self.block_tables:
        block_ids = self.block_tables[request_id]
        self.free_block_list.extend(block_ids)  # Return to free pool
        self.free_block_list.sort()  # Maintain sorted order
        del self.block_tables[request_id]  # Remove tracking entry
        
        # Optionally clear tensors to free GPU memory
        for block_id in block_ids:
            self.blocks[block_id].keys = None
            self.blocks[block_id].values = None
```

**Key Aspects:**
- Returns blocks to the free list for reuse
- Maintains sorted order for allocation efficiency
- Removes request tracking information
- Optionally clears GPU tensors to free memory

### 3. Block Copying

For advanced operations like request preemption or memory defragmentation:

```python
def copy_blocks(self, src_block_ids: List[int], dst_block_ids: List[int]):
    if len(src_block_ids) != len(dst_block_ids):
        raise ValueError("Source and destination lists must have same length")
    
    for src_id, dst_id in zip(src_block_ids, dst_block_ids):
        src_block = self.blocks[src_id]
        dst_block = self.blocks[dst_id]
        
        # Allocate destination if needed
        if dst_block.keys is None or dst_block.values is None:
            dst_block.allocate()
        
        # Copy data
        with torch.no_grad():
            dst_block.keys.copy_(src_block.keys)
            dst_block.values.copy_(src_block.values)
```

## Memory Layout and Access

### Block Organization

The memory is organized as a collection of fixed-size blocks:

```
Global Memory Pool:
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │ Block 5 │
│ [16xHxD]│ [16xHxD]│ [16xHxD]│ [16xHxD]│ [16xHxD]│ [16xHxD]│
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

Request A: Uses [Block 0, Block 2] for non-contiguous sequence storage
Request B: Uses [Block 1, Block 4, Block 5] for its sequence
```

Where H = num_heads and D = head_dim

### Block Tables

Each request has an associated block table that maps logical token positions to physical blocks:

```
Request A Block Table:
Logical: [0-15][16-31][32-47][48-63][64-79]
Physical: Block0 Block2  -     -     Block4
```

## Integration with SGLang Features

### Computation Sharing

The paged system enables computation sharing by allowing requests with shared prefixes to reference the same memory blocks:

- Requests with common prefixes can share the same KV-cache blocks
- Multiple requests can reference the same physical memory location
- Reduces redundant computation and memory usage

### Memory Efficiency

By using fixed-size blocks:

- Memory fragmentation is eliminated
- Block reuse is maximized
- Memory utilization approaches optimal levels

## Performance Considerations

### Block Size Selection

The block size parameter is critical for performance:

- **Smaller blocks**: Less internal fragmentation, more overhead for block management
- **Larger blocks**: More internal fragmentation, less management overhead
- **Typical values**: 8-32 tokens per block work well in practice

### Memory Allocation Strategy

The system uses a simple first-fit strategy:

```python
# Allocate from beginning of free list
block_id = self.free_block_list.pop(0)
```

For production systems, more sophisticated strategies might include:
- Best-fit to minimize fragmentation
- Coalescing strategies to combine blocks
- Preallocation to reduce allocation overhead

## Memory Safety and Management

### GPU Memory Management

The system ensures proper GPU memory allocation:

```python
def allocate(self):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.keys = torch.zeros(
        self.size, self.num_heads, self.head_dim, 
        dtype=self.dtype, device=device
    )
    self.values = torch.zeros(
        self.size, self.num_heads, self.head_dim, 
        dtype=self.dtype, device=device
    )
```

### Memory Cleanup

Proper cleanup prevents memory leaks:

- Free blocks when requests complete
- Clear GPU tensors to release memory
- Maintain consistent state in block tables

## Advanced Features

### Dynamic Block Resizing

For requests that need to extend beyond their initial allocation:

- Allocate additional blocks as needed
- Maintain logical sequence continuity
- Update block tables accordingly

### Memory Pool Management

Advanced implementations might include:

- Block migration to reduce fragmentation
- Eviction policies for memory-constrained scenarios
- Prefetching strategies for better performance

## Error Handling

### Out of Memory Conditions

The system handles memory exhaustion gracefully:

```python
if len(self.free_block_list) < num_blocks_needed:
    raise RuntimeError(f"Not enough free blocks. Need {num_blocks_needed}, have {len(self.free_block_list)}")
```

### Block Validation

Before operations, the system validates block states:

- Verify blocks are allocated before accessing
- Check for proper tensor dimensions
- Validate request associations

## Future Enhancements

### Memory Optimization

Potential improvements include:

- Compressed KV-cache storage
- Offloading to CPU memory when possible
- Cache eviction policies for long-running requests

### Performance Optimization

Advanced techniques might include:

- Block prefetching for better cache performance
- Heterogeneous memory management (different memory types)
- Asynchronous memory operations

## Implementation Variations

### SGLang-Style Memory Management

For SGLang-specific optimizations:

- Prefix sharing memory management
- Radix tree integration for shared computation
- Advanced scheduling based on memory access patterns

### Integration Points

The memory manager connects with other components:

- Scheduler: Provides memory availability information
- Attention modules: Access KV-cache through block tables
- Model execution: Uses paged cache for efficient attention computation