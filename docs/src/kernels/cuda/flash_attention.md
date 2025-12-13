# Flash Attention (`kernels/cuda/flash_attention.cu`)

## 1. Concept: Memory Bandwidth

The main bottleneck in Attention is reading the huge $N \times N$ matrix from memory.
**Flash Attention** breaks the problem into small "tiles" that fit into the GPU's fast **SRAM** (Shared Memory). We compute everything for that tile without going back to slow Global Memory.

```mermaid
graph TB
    subgraph GlobalMemory [Global Memory (HBM)]
        Q[Matrix Q]
        K[Matrix K]
        V[Matrix V]
    end

    subgraph SRAM [Shared Memory (SRAM)]
        TileQ[Tile Q]
        TileK[Tile K]
        TileV[Tile V]
        Comp(("Compute QK^T * V"))
    end

    Q --> TileQ
    K --> TileK
    V --> TileV

    TileQ --> Comp
    TileK --> Comp
    TileV --> Comp
```

---

## 2. Implementation Guide

We will implement a **simplified** version. Doing full FlashAttention v2 is extremely complex. We aim for "Tiled Attention".

### Step 0: The Setup

Open `src/kernels/cuda/flash_attention.cu`.
Identify the `flash_attention_forward` function.

You have pointers to:

- `query` (Q), `key` (K), `value` (V) residing in Global Memory.

### Step 1: Define Thread Layout

We want to process tiles.

- **Grid**: One block per query chunk.
- **Block**: Threads within the block handle individual heads or elements.

```cpp
// Example
dim3 grid(num_batches, num_heads);
dim3 block(128); // 128 threads work together on one head
```

### Step 2: Load Tiles to Shared Memory

You need `__shared__` memory arrays.

```cpp
__shared__ float s_Q[TILE_SIZE][HEAD_DIM];
__shared__ float s_K[TILE_SIZE][HEAD_DIM];
```

Use `threadIdx.x` to cooperatively load data from Global `Q` to Shared `s_Q`.
**Remember**: call `__syncthreads()` after loading!

### Step 3: Compute $QK^T$ (Scores)

Iterate over your shared Q and K.
Calculate the dot product.
Store in a register (local variable).

### Step 4: Softmax (The "Online" Trick)

In standard softmax, you need the max of the _entire_ row. Here we only see a tile!
**Trick**: Keep a running max ($m$) and running sum ($l$). Update them as you see new tiles.

- $m_{new} = \max(m_{old}, \max(current\_tile))$
- Adjust previous sums by multiplying by $e^{m_{old} - m_{new}}$.

### Step 5: Compute Score $\times$ V

Once you have the probabilities for the tile, multiply by `s_V` (which you also loaded).
Accumulate into `output`.

---

## 3. Hints

- Start with a **Naive** kernel first! Forget shared memory. Just loops.
  - Thread per query token.
  - Loop over all key tokens.
  - Compute.
  - This is $O(N^2)$ memory reads but verifies your logic is correct.
- Only optimize to Shared Memory once logic works.
