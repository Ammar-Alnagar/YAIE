# Welcome to Mini-YAIE

**Mini-YAIE** (Yet Another Inference Engine) is an educational project designed to demystify modern Large Language Model (LLM) inference engines.

Driven by the need for efficiency, modern engines like **SGLang**, **vLLM**, and **TensorRT-LLM** use sophisticated techniques to maximize GPU throughput and minimize latency. Mini-YAIE provides a simplified, clean implementation of these concepts, focusing on:

- **Continuous Batching**
- **Paged KV Caching**
- **Radix Attention (Prefix Sharing)**

## How to use this guide

This documentation is structured to take you from high-level concepts to low-level implementation.

1.  **Core Concepts**: Start here to understand the _why_ and _what_ of inference optimization.
2.  **Architecture**: Understand how the system components fit together.
3.  **Implementation Guides**: Step-by-step guides to implementing the missing "kernels" in Python and CUDA.

## Your Mission

The codebase contains **placeholders** (`NotImplementedError`) for critical components. Your goal is to implement these components following this guide, turning Mini-YAIE from a skeleton into a fully functional inference engine.
