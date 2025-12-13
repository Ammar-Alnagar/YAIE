"""
SGLang-style configuration module
Contains configuration settings for SGLang-style inference
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SGLangConfig:
    """Configuration for SGLang-style inference engine"""

    # Scheduler settings
    max_batch_size: int = 8
    max_prefill_batch_size: int = 16
    max_decode_batch_size: int = 256
    max_seq_len: int = 2048

    # KV Cache settings
    num_gpu_blocks: int = 2000
    num_cpu_blocks: int = 1000
    block_size: int = 16

    # Model settings
    dtype: str = "float16"  # or "float32", "bfloat16"
    tensor_parallel_size: int = 1

    # Memory settings
    gpu_memory_utilization: float = 0.9  # How much GPU memory to use
    swap_space: int = 4  # GB to use for CPU swap

    # Generation settings
    default_max_tokens: int = 1024
    default_temperature: float = 1.0
    default_top_p: float = 1.0

    # SGLang-specific features
    enable_radix_cache: bool = True  # Enable radix attention cache
    enable_chunked_prefill: bool = True  # Enable chunked prefill for long prompts
    schedule_policy: str = "fcfs"  # "fcfs", "priority", etc.

    # Prefix caching settings
    enable_prefix_caching: bool = True
    max_num_schedule_steps: int = 1000  # Maximum steps before preemption

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary"""
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }


# Default configuration
SGLANG_DEFAULT_CONFIG = SGLangConfig()


def get_sglang_config(**overrides) -> SGLangConfig:
    """Get SGLang configuration with optional overrides"""
    config = SGLANG_DEFAULT_CONFIG
    if overrides:
        # Create a new instance with overrides
        config_dict = config.to_dict()
        config_dict.update(overrides)
        config = SGLangConfig.from_dict(config_dict)
    return config
