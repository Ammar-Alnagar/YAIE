# KV Cache Manager (`kernels/kv_cache.py`)

## 1. Concept: Paged Attention

In a standard implementation, KV Cache is a huge contiguous tensor `[MAX_SEQ_LEN, HEADS, DIM]`. This wastes memory because most prompts are shorter than `MAX_SEQ_LEN`.

**Paged Attention** breaks this tensor into small fixed-size blocks (e.g., size 16).

- **Physical Memory**: A big pool of blocks `[NUM_BLOCKS, 16, HEADS, DIM]`.
- **Logical Memory**: For each request, we just keep a list of block indices `[0, 5, 12]`.

Your job is to write the **Allocator** (like `malloc` in C).

---

## 2. Implementation Guide

Open `src/kernels/kv_cache.py`.

### Step 1: Initialization

We need to track which blocks are free and which are used.

**Task**: In `__init__`:

1.  Create a list `self.free_blocks`. Initially, it should contain _all_ integers from `0` to `num_blocks - 1`.
2.  Create a dictionary `self.block_tables`. This will map `request_id -> List[int]` (the list of blocks owned by that request).

```python
# Hint
self.free_blocks = list(range(num_blocks))
```

---

### Step 2: The `allocate_blocks` Method

When a request comes in (or generates new tokens), it needs memory.

**Signature**:

```python
def allocate_blocks(self, request_id: str, num_tokens: int) -> List[int]:
```

**Algorithm**:

1.  Calculate how many blocks are needed.
    - $N_{blocks} = \lceil num\_tokens / block\_size \rceil$
2.  Check if we have enough `free_blocks`.
    - If `len(free_blocks) < needed`, raise an Error (or handle OOM).
3.  **Pop** the blocks from `free_blocks`.
4.  **Assign** them to `self.block_tables[request_id]`.
5.  Return the list of allocated block indices.

**Your Turn**: Implement this logic. Watch out for integer division!

---

### Step 3: The `free_blocks` Method

When a request finishes, we must reclaim memory.

**Algorithm**:

1.  Look up the blocks for `request_id`.
2.  **Append** them back to `self.free_blocks`.
3.  Delete the entry from `self.block_tables`.

**Critical**: Do not double-free! (Though Python sets make this easier, a list is faster for standard stacks).

---

### Step 4: Connecting to the Engine

The `get_kv_tensors` method is checking if you can translate the "Logical" view to the "Physical" view.

**Task**: Implement `get_kv_tensors`.

- It should presumably return the specific GPU tensors for the blocks.
- _Note_: In this Python simulation, just returning the indices is often enough for the Scheduler to know mapping. The actual _Tensor_ access happens in the CUDA kernel.

---

### Step 5: Verify

Create `tests/test_kv_manual.py`:

```python
manager = KVCacheManager(num_blocks=10, block_size=16, ...)
# Alloc 20 tokens -> needs 2 blocks (indices 0, 1)
blocks = manager.allocate_blocks("req1", 20)
print(blocks)
# Free
manager.free_blocks("req1")
print(len(manager.free_blocks)) # Should be 10 again
```
