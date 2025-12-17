# Configuration Management

## Overview

Mini-YAIE uses a flexible configuration system that allows users to customize various aspects of the inference engine without modifying the code. The configuration system provides settings for memory management, scheduling, model loading, and performance optimization.

## Configuration Structure

The configuration system is built around the `SGLangConfig` dataclass in `src/config.py`. The system supports:

- Dataclass-based configuration with type hints
- Default values for all parameters
- Dictionary-based overrides
- Component-specific configuration sections

## Key Configuration Parameters

### Scheduler Configuration

```python
# Maximum batch size for processing requests
max_batch_size: int = 8

# Maximum batch size for prefill operations
max_prefill_batch_size: int = 16

# Maximum batch size for decode operations
max_decode_batch_size: int = 256

# Maximum sequence length allowed
max_seq_len: int = 2048
```

### KV Cache Configuration

```python
# Number of GPU memory blocks for KV-cache
num_gpu_blocks: int = 2000

# Number of CPU memory blocks for swapping
num_cpu_blocks: int = 1000

# Size of each memory block (in tokens)
block_size: int = 16
```

### Model Configuration

```python
# Data type for model weights and KV-cache
dtype: str = "float16"  # Options: "float16", "float32", "bfloat16"

# Tensor parallelism size
tensor_parallel_size: int = 1

# GPU memory utilization fraction
gpu_memory_utilization: float = 0.9

# CPU swap space in GB
swap_space: int = 4
```

### Generation Configuration

```python
# Default maximum tokens to generate per request
default_max_tokens: int = 1024

# Default sampling temperature
default_temperature: float = 1.0

# Default top-p value
default_top_p: float = 1.0
```

### SGLang-Specific Features

```python
# Enable radix attention cache for prefix sharing
enable_radix_cache: bool = True

# Enable chunked prefill for long prompts
enable_chunked_prefill: bool = True

# Scheduling policy: "fcfs" (first-come-first-served)
schedule_policy: str = "fcfs"

# Enable prefix caching
enable_prefix_caching: bool = True

# Maximum scheduling steps before preemption
max_num_schedule_steps: int = 1000
```

## Configuration Loading

### Default Configuration

When no explicit configuration is provided, the system uses sensible defaults that work well for most educational purposes:

- Conservative memory usage to work on most GPUs
- Balanced performance settings
- Safe batch sizes that avoid out-of-memory errors

### Custom Configuration

Users can customize configurations by:

1. **Direct parameter passing** to constructors
2. **Environment variables** for deployment scenarios
3. **Configuration files** (when implemented)

## Configuration Best Practices

### Performance Tuning

For production use, consider these configuration adjustments:

- **Increase batch sizes** based on available GPU memory
- **Adjust block size** for optimal cache utilization
- **Tune memory pool size** based on request patterns

### Memory Management

Configure memory settings based on your hardware:

```python
# For high-end GPUs (24GB+ VRAM)
num_blocks = 4000
max_batch_size = 32

# For mid-range GPUs (8-16GB VRAM)
num_blocks = 1000
max_batch_size = 8

# For entry-level GPUs (4-8GB VRAM)
num_blocks = 500
max_batch_size = 4
```

## Integration with Components

### Engine Integration

The main engine uses the SGLangConfig for initialization:

```python
from src.config import SGLangConfig, get_sglang_config

# Use default config
config = get_sglang_config()

# Or override specific parameters
config = get_sglang_config(
    max_batch_size=16,
    num_gpu_blocks=4000
)

# Initialize components with config values
scheduler = SGLangScheduler(
    max_batch_size=config.max_batch_size,
    max_prefill_batch_size=config.max_prefill_batch_size,
    max_decode_batch_size=config.max_decode_batch_size
)
```

### Scheduler Configuration

The SGLang scheduler uses configuration for scheduling policies:

- Batch size limits
- Prefill/decode phase sizing
- Memory-aware scheduling decisions

### Memory Manager Configuration

The KV-cache manager uses configuration for:

- Total memory pool size
- Block allocation strategies
- Memory optimization policies

## Environment-Specific Configuration

### Development Configuration

For development and learning:

- Conservative memory limits
- Detailed logging
- Debug information enabled

### Production Configuration

For production deployment:

- Optimized batch sizes
- Performance-focused settings
- Minimal logging overhead

## Configuration Validation

The system validates configuration parameters to prevent:

- Memory allocation failures
- Invalid parameter combinations
- Performance-degrading settings

## Future Extensions

The configuration system is designed to accommodate:

- Model-specific optimizations
- Hardware-aware tuning
- Runtime configuration updates
- Performance auto-tuning

## Configuration Examples

### Basic Configuration
```python
# Minimal configuration for learning
config = {
    "max_batch_size": 4,
    "num_blocks": 1000,
    "block_size": 16
}
```

### Performance Configuration
```python
# Optimized for throughput
config = {
    "max_batch_size": 32,
    "max_decode_batch_size": 512,
    "num_blocks": 4000,
    "block_size": 32
}
```

## Troubleshooting Configuration Issues

### Memory Issues

If experiencing out-of-memory errors:
1. Reduce `num_blocks` in KV-cache
2. Lower batch sizes
3. Check available GPU memory

### Performance Issues

If experiencing low throughput:
1. Increase batch sizes
2. Optimize block size for your model
3. Verify CUDA availability and compatibility