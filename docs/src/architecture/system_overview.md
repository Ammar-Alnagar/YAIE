# System Architecture Overview

## Introduction

Mini-YAIE (Yet Another Inference Engine) is an educational implementation of modern LLM inference techniques, specifically designed to demonstrate concepts from state-of-the-art systems like SGLang, vLLM, and TensorRT-LLM. The architecture focuses on three core optimizations:

1. **Continuous Batching**: Dynamically batching incoming requests to maximize GPU utilization
2. **Radix Attention**: Efficient attention mechanism with prefix sharing for similar requests
3. **Paged KV-Cache**: Memory-efficient key-value cache management

## High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Layer     │    │  Engine Core    │    │  Model/Kernels  │
│  (FastAPI)      │◄──►│  (Scheduler,   │◄──►│  (PyTorch/     │
│                 │    │  Attention)    │    │  CUDA)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Layer     │    │  Model Loading  │    │  Memory Mgmt    │
│  (yaie serve/   │    │  (HuggingFace  │    │  (Paged Cache)  │
│   yaie chat)    │    │  Integration)   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Components

### 1. Main Inference Engine (`engine.py`)

The main inference engine orchestrates all components and provides the high-level API for inference. It implements SGLang-style continuous batching with radix attention and prefix sharing.

**Key Responsibilities:**
- Request orchestration and management
- Integration between scheduler, attention mechanisms, and memory management
- API layer communication
- Model loading and tokenizer management

### 2. SGLang Scheduler (`core/sglang_scheduler.py`)

The SGLang-style scheduler implements advanced request scheduling with:
- **Prefix-based request grouping**: Groups requests with common prefixes for computation sharing
- **Separate prefill and decode scheduling**: Optimizes for the different computational patterns
- **Memory-aware batch sizing**: Considers available KV-cache memory when scheduling
- **Continuous batching optimization**: Maintains high GPU utilization

### 3. Radix Attention System (`kernels/radix_attention.py`)

Implements the radial attention mechanism with:
- **Prefix sharing**: Reduces redundant computation for requests with common prefixes
- **Paged KV-cache integration**: Efficient memory management for variable-length requests
- **RoPE (Rotary Position Embeddings)**: Supports position-aware attention

### 4. Paged KV-Cache Management (`kernels/kv_cache.py`)

Efficient memory management using page-based allocation:
- **Fixed-size blocks**: Reduces memory fragmentation
- **Request-to-block mapping**: Tracks which blocks belong to which requests
- **Dynamic allocation/deallocation**: Manages memory based on request lifecycle

### 5. Radix Tree System (`kernels/radix_tree.py`)

Enables efficient prefix matching and computation sharing:
- **Trie-based structure**: Organizes token sequences hierarchically
- **Request grouping**: Identifies requests with shared prefixes
- **Computation optimization**: Provides information for scheduler optimization

### 6. Sampling Kernel (`kernels/sampling.py`)

Implements core sampling algorithms:
- **Temperature scaling**: Controls randomness in generation
- **Top-K sampling**: Limits selection to top K most probable tokens
- **Top-P (Nucleus) sampling**: Limits selection to tokens that sum to probability P

### 7. API Server (`server/api.py`)

Provides OpenAI-compatible API endpoints:
- **RESTful design**: Follows OpenAI's API specification
- **Streaming support**: Real-time token streaming
- **Health monitoring**: Server status endpoints

## Data Flow

The system processes requests in the following sequence:

1. **Request Arrival**: Client sends a request through the API layer
2. **Request Scheduling**: SGLang scheduler groups requests with common prefixes
3. **Prefill Phase**: Process full prompt sequences using radial attention
4. **Decode Phase**: Generate tokens one-by-one with shared computation
5. **KV-Cache Management**: Efficient memory allocation and sharing
6. **Response Generation**: Return results via API layer

## Key Design Principles

### Modularity
Each component is designed to be independent, allowing for focused learning and experimentation.

### Educational Focus
Clean, well-documented code with comprehensive explanations of key concepts.

### SGLang-Style Optimization
Focus on prefix sharing and radix trees to maximize computational efficiency.

### Memory Efficiency
Paged cache management to reduce memory fragmentation and maximize utilization.

## Architecture Benefits

1. **High Throughput**: Continuous batching and prefix sharing maximize GPU utilization
2. **Memory Efficiency**: Paged KV-cache reduces fragmentation and enables larger batch sizes
3. **Scalability**: Modular design allows for optimization of individual components
4. **Educational Value**: Clean implementation of state-of-the-art techniques

## Integration Points

The system integrates components through well-defined interfaces:

- Engine connects to scheduler for request management
- Scheduler connects to memory manager for KV-cache coordination
- Attention mechanisms access KV-cache through the memory manager
- Sampler provides token selection for generation
- API layer communicates with the engine for request processing