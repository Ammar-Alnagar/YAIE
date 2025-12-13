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

```mermaid
graph LR
    subgraph Logical["Logical Sequence (Request)"]
        L0[Block 0: "Hello"]
        L1[Block 1: "World"]
        L2[Block 2: "!"]
    end

    subgraph Table["Page Table"]
        T0[0 -> 7]
        T1[1 -> 2]
        T2[2 -> 9]
    end

    subgraph Physical["GPU Memory (Physical Blocks)"]
        B0[Block 0]
        B1[Block 1]
        B2[Block 2: "World"]:::used
        B3...
        B7[Block 7: "Hello"]:::used
        B8...
        B9[Block 9: "!"]:::used
    end

    L0 --> T0 --> B7
    L1 --> T1 --> B2
    L2 --> T2 --> B9

    classDef used fill:#aaffaa;
```

## The Kernel

The Paged Attention kernel allows the Attention mechanism to read keys and values from these non-contiguous blocks on the fly, enabling near-zero memory waste.
