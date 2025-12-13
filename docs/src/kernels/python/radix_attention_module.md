# Radix Attention Module (`kernels/radix_attention.py`)

## 1. Concept: Connecting the Dots

We have a **Radix Tree** (prefix matching) and a **Paged KV Cache** (memory management). The `RadixAttentionWithPagedKVCache` class is the glue that runs on the CPU (Python side) to manage these resources before we launch the GPU kernels.

It doesn't run the attention _math_ (that's the CUDA kernel's job). Instead, it manages the **metadata**:

- "Request A needs to append 'cat' to its sequence."
- "Does 'cat' already exist in the Radix Tree?"
- "If yes, reuse the block."
- "If no, allocate a new block."

---

## 2. Implementation Guide

Open `src/kernels/radix_attention.py`.

### Step 1: Initialization

You need to initialize the two sub-components we built earlier.

```python
class RadixAttentionWithPagedKVCache:
    def __init__(self, ...):
        # ...
        self.radix_tree = RadixTree()
        self.kv_cache_manager = KVCacheManager(...)
```

### Step 2: `append_slot` (The Critical Logic)

This method is called when we want to add a new token (or tokens) to a request.

**Signature**:

```python
def append_slot(self, key: torch.Tensor, value: torch.Tensor, request_id: str):
```

- `key`/`value`: The computed K/V tensors for the _new_ token(s).

**Algorithm**:

1.  **Check Tree**: Use `self.radix_tree` to see if this `(request_id + new_token)` path already exists?
    - _Note_: In a real system, we check _before_ computing K/V. Here, we might just be managing the cache storage.
2.  **Allocate**: If we need new space, call `self.kv_cache_manager.allocate_blocks()`.
3.  **Store**: We need to perform the copy.
    - Ideally, we just return the _indices_ of where to write, and the GPU kernel does the writing.
    - For this Python simulation, you might simulate the copy or just track the metadata.

### Step 3: `get_kv_cache`

The scheduler asks: "I am about to run requests `[R1, R2]`. Where is their data?"

**Algorithm**:

1.  Loop through `request_ids`.
2.  For each, ask `self.kv_cache_manager` for its block table (list of integers).
3.  Pack these lists into a single Tensor `block_tables`.
4.  Return `block_tables` to the Engine.

### Step 4: `free_request`

When a request is done:

1.  `self.radix_tree.remove_request(request_id)` (Decrement ref counts).
2.  `self.kv_cache_manager.free_blocks(request_id)` (Reclaim memory).

---

## 3. The `RadixAttentionBlock` (Model Layer)

The class `RadixAttentionBlock` is the PyTorch module that sits in the model.

**Task**:
In `forward()`:

1.  Compute Q, K, V projections.
2.  Compute RoPE (Rotary Embeddings).
3.  **If Prefill**: Use Flash Attention (or a standard attention) on the new tokens.
4.  **If Decode**:
    - Call `append_slot` to save the new K/V.
    - Call `paged_attention_kernel` (the CUDA op) to attend to the _entire_ history using the block tables.

**Exercise**:
Since we don't have the full model weight loading for this specific block, focus on the **logic flow** in the comments.
