# Flash Attention (`kernels/cuda/flash_attention.cu`)

## Concept

A standard attention implementation is $O(N^2)$ in memory usage. Flash Attention uses tiling to compute attention in constant memory.

## Implementation Goal

Implement `flash_attention_forward`:

### Algorithm (simplified)

1.  **Tiling**: Load a block of Queries (Q) into shared memory (SRAM).
2.  **Loop**: Iterate over blocks of Keys (K) and Values (V) from HBM (Global Memory).
    - Load K, V block into SRAM.
    - Compute QK^T (Attention Scores).
    - Apply Softmax (using online softmax scaling).
    - Compute Score \* V.
    - Accumulate result to Output.
3.  **Write Output**: Store final result to HBM.

> **Note**: For this educational project, a naive CUDA implementation is acceptable if tiling is too complex.
