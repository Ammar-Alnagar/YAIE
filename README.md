<div align="center">
  <img src="logo.png" alt="YAIE Logo" width="200" height="200">

  <!-- Note: Add your logo to the logo.png file in this directory -->
</div>

# YAIE: Educational LLM Inference Engine

YAIE (Yet Another Inference Engine) is an educational project designed to help students and developers understand how modern LLM inference engines work. This implementation is inspired by state-of-the-art systems like SGLang, vLLM, FlashInfer and other efficient inference engines, focusing on continuous batching, radial attention, and FlashInfer-style optimizations.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Learning Objectives](#learning-objectives)
- [Implementation Guide](#implementation-guide)
- [Building Kernels](#building-kernels)
- [Contributing](#contributing)
- [License](#license)

## Overview

Modern LLM inference engines like SGLang, vLLM, and TensorRT-LLM implement sophisticated techniques to maximize throughput and minimize latency. YAIE demonstrates these concepts through a simplified but educational implementation that focuses on:

- **Continuous Batching**: Dynamically batching incoming requests to maximize GPU utilization
- **Radial Attention**: Efficient attention mechanism with prefix sharing and paged KV-cache
- **OpenAI Compatibility**: Server mode provides OpenAI-compatible API
- **Modular Design**: Clean architecture separating concerns for easy learning

## Features

- **Two Operation Modes**:
  - Server mode (`yaie serve`) with OpenAI-compatible API
  - CLI chat mode (`yaie chat`) for interactive conversations
- **HuggingFace Integration**: Automatic model downloading and caching
- **Continuous Batching**: Efficient request scheduling for better throughput
- **Paged KV-Cache**: Memory-efficient key-value cache management
- **Radial Attention**: Prefix sharing for similar requests
- **Educational Focus**: Clear, well-documented code with learning resources

## Architecture

The engine follows a modular architecture:

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

### Key Components:

1. **CLI Interface**: Entry point for both server and chat modes
2. **API Server**: FastAPI-based server with OpenAI-compatible endpoints
3. **Inference Engine**: Core processing logic with scheduler and attention
4. **Scheduler**: Continuous batching with request management
5. **Radial Attention**: Efficient attention with prefix sharing
6. **Model Loader**: HuggingFace model and tokenizer management
7. **KV-Cache Manager**: Paged cache for efficient memory usage

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/YAIE.git
   cd YAIE
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package:
   ```bash
   pip install -e .
   ```

## Usage

### Server Mode (OpenAI API Compatible)

Start the server with a specific model:
```bash
yaie serve microsoft/DialoGPT-medium --host localhost --port 8000
```

The server will:
- Check for the model in local HuggingFace cache
- Download if not present
- Start an OpenAI-compatible API server

API endpoints:
- `POST /v1/chat/completions` - Chat completions
- `GET /v1/models` - List available models

### CLI Chat Mode

Start an interactive chat session:
```bash
yaie chat microsoft/DialoGPT-medium
```

The chat will:
- Check for the model in local HuggingFace cache
- Download if not present
- Start an interactive chat session

### Building Kernels

To build the custom CUDA kernels for optimized performance:

```bash
# Using the build script
./build_kernels.sh

# Or using make
make build-kernels

# Or directly with Python
python setup_kernels.py build_ext --inplace
```

**Note**: Kernel building requires:
- CUDA toolkit installed
- PyTorch with CUDA support
- Compatible GPU with compute capability >= 6.0

If CUDA is not available, the engine will run in CPU-only mode.

## Learning Objectives

By implementing the kernels and components in this project, you will learn:

1. **Continuous Batching Concepts**:
   - Problem: Traditional batching requires all requests to have the same length
   - Solution: Dynamically batch requests and handle them at different stages

2. **Paged KV-Cache Management**:
   - Problem: KV-cache memory fragmentation with variable-length requests
   - Solution: Use paged memory management similar to OS virtual memory

3. **Radial Attention & Prefix Sharing**:
   - Problem: Redundant computation for requests with similar prefixes
   - Solution: Share computed attention across requests with common prefixes (SGLang-style)

4. **FlashInfer-Style Optimizations**:
   - Problem: Inefficient memory access patterns during attention computation
   - Solution: Optimized attention for both prefill and decode phases

5. **CUDA Kernel Programming**:
   - Efficient GPU memory access patterns
   - Parallel computation for attention mechanisms
   - Memory bandwidth optimization

6. **System Performance Optimization**:
   - Latency vs. throughput trade-offs
   - Memory management strategies
   - Batch size optimization

## Implementation Guide

This project provides a detailed guide for implementing the various kernels and components:

- **[Implementation Guide](guide.md)**: Complete documentation of all kernels that need to be implemented

### Kernels to Implement:

1. **Attention Kernels**:
   - FlashAttention forward and backward
   - Paged attention
   - RoPE (Rotary Position Embedding)

2. **Normalization Kernels**:
   - RMS normalization

3. **Activation Kernels**:
   - SiLU and multiplication fusion

4. **Memory Management**:
   - KV-cache management with paging
   - Block allocation and deallocation

5. **CPU Fallbacks**:
   - CPU implementations for when GPU is not available

## Contributing

This is an educational project, and contributions that improve the learning experience are welcome:

1. Add more detailed comments explaining complex concepts
2. Create additional examples or tutorials
3. Improve documentation and explanations
4. Add more model compatibility

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by SGLang, vLLM, and other efficient inference engines
- Built on top of HuggingFace Transformers library
- Educational reference implementation for learning purposes

We would like to acknowledge the significant contributions of several state-of-the-art inference engines that inspired this educational project:

- **vLLM**: For pioneering the concept of PagedAttention and efficient memory management in LLM serving
- **SGLang**: For introducing radial attention and highly optimized prompt processing techniques
- **TensorRT-LLM**: For demonstrating the power of optimized inference through NVIDIA's TensorRT technology
- **LightLLM**: For showing how to implement efficient inference with various optimization techniques

These projects have advanced the field of LLM inference significantly, and this educational engine draws concepts and inspiration from their innovative approaches to continuous batching, attention optimization, and memory management.
