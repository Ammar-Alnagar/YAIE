# Paged Attention (vLLM)

**Paged Attention** is the core innovation of vLLM. It optimizes the **Decode Phase** by managing memory like an Operating System.

## The Problem: Memory Fragmentation

Before vLLM, engines allocated contiguous memory for the maximum possible length of a request.

- **Internal Fragmentation**: If a request was shorter than max length, memory was wasted.
- **External Fragmentation**: We couldn't fit a new request even if total free memory was sufficient, because no single contiguous block was large enough.

## The Solution: Pagiing

Inspired by virtual memory in OS:

1.  **Blocks**: Divide KV Cache into fixed-size blocks (e.g., 16 tokens per block).
2.  **Non-Contiguous**: Blocks can be stored anywhere in physical GPU memory.
3.  **Mapping**: A "Block Table" maps logical token positions to physical block addresses.

## The Kernel

The Paged Attention kernel allows the Attention mechanism to read keys and values from these non-contiguous blocks on the fly, enabling near-zero memory waste.
