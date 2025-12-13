# Python Kernels Guide

This section guides you through implementing the core Python logic "kernels". These are not CUDA kernels, but critical algorithmic components.

## Your Tasks

1.  **Radix Tree**: Implement the Trie data structure for prefix matching.
2.  **KV Cache Manager**: Implement the block allocation strategy.
3.  **Sampling**: Implement the token sampling logic.

**Why Python?**
While computation happens in CUDA, the _logic_ of memory management and prefix matching is complex and best handled in Python (or C++ CPU code) before dispatching to the GPU.
