# Summary

- [Introduction](intro/welcome.md)

   - [Prerequisites](intro/prerequisites.md)
   - [Environment Setup](intro/setup.md)
   - [Model Loading](intro/model_loading.md)

- [Core Concepts](concepts/llm_inference.md)

  - [Continuous Batching](concepts/continuous_batching.md)
  - [Radix Attention (SGLang)](concepts/radix_attention.md)
  - [Paged Attention (vLLM)](concepts/paged_attention.md)

- [Architecture Deep Dive](architecture/system_overview.md)

   - [Configuration Management](architecture/configuration.md)
   - [Engine Orchestration](architecture/engine.md)
   - [Scheduler Logic](architecture/scheduler.md)
   - [Memory Management](architecture/memory_manager.md)

- [Python Kernels Guide](kernels/python/overview.md)

  - [Radix Tree (Trie)](kernels/python/radix_tree.md)
  - [KV Cache Manager](kernels/python/kv_cache_manager.md)
  - [Radix Attention Module](kernels/python/radix_attention_module.md)
  - [Sampling Logic](kernels/python/sampling_module.md)

- [CUDA Kernels Guide](kernels/cuda/setup.md)

  - [Memory Operations](kernels/cuda/memory_ops.md)
  - [Flash Attention](kernels/cuda/flash_attention.md)
  - [Paged Attention](kernels/cuda/paged_attention.md)
  - [Radix Operations](kernels/cuda/radix_ops.md)
  - [Implementation Guide](kernels/cuda/implementation_guide.md)

- [API & Serving](serving/api_endpoints.md)

  - [CLI Usage](serving/cli.md)
  - [Production Deployment](serving/production.md)

- [Appendices](appendix/references.md)
  - [Troubleshooting](appendix/troubleshooting.md)
