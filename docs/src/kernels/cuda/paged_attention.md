# Paged Attention (`kernels/cuda/paged_attention.cu`)

## 1. Concept: Indirection

Paged Attention is just standard attention, but `K` and `V` are not contiguous.
We have to "gather" them using a Page Table.

```mermaid
graph LR
    Thread -->|1. Get Logical idx| Logic[Token #42]
    Logic -->|2. Lookup Table| Table[Block 2, Offset 10]
    Table -->|3. Get Physical Addr| Phys[0xA000...]
    Phys -->|4. Read| Data[Value]
```

---

## 2. Implementation Guide

### Step 1: Understand the Block Table

You are passed `block_tables` tensor of shape `[num_seqs, max_blocks]`.

- It holds integer indices of physical blocks.
- `block_tables[req_id][0]` is the first block of that request.

### Step 2: Calculate Physical Address

Inside your kernel, you want the Key vector for token `t` of request `r`.

```cpp
int block_idx = t / BLOCK_SIZE;
int block_offset = t % BLOCK_SIZE;
int physical_block_number = block_tables[r][block_idx];

// Pointer arithmetic
float* k_ptr = key_cache_base
             + physical_block_number * (BLOCK_SIZE * HEAD_DIM * NUM_HEADS)
             + ... // navigate to specific head and offset
```

### Step 3: Load Data

Using the pointer `k_ptr`, load the vector into registers or shared memory.

### Step 4: Compute Attention

Once loaded, the math is identical to standard Attention or Flash Attention.
$Q \cdot K^T$, Softmax, $\cdot V$.

---

## 3. Your Task

Implement `paged_attention_kernel` in `src/kernels/cuda/paged_attention.cu`.

1.  Focus on the **address calculation** logic. That is the only difference!
2.  Use the `copy_blocks` kernel (Memory Ops) to help set up test data if needed.
