"""
Setup script for building custom CUDA kernels for YAIE
This script handles compilation of custom attention and other kernels
"""

import os

import torch
from torch.utils.cpp_extension import CUDAExtension, setup


def get_cuda_arch_flags():
    """Get CUDA architecture flags based on available GPU"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA kernel compilation")
        return []

    # Get the compute capability of the first available GPU
    device = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch_flag = f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"

    print(f"Building CUDA kernels for {device} (Compute Capability: {major}.{minor})")
    return [arch_flag]


def build_extensions():
    """Build all CUDA extensions"""
    extensions = []

    # Only build if CUDA is available
    if torch.cuda.is_available():
        arch_flags = get_cuda_arch_flags()

        # FlashAttention extension
        extensions.append(
                name="mini_yaie_kernels.flash_attention",
                sources=[
                    "src/kernels/cuda/flash_attention.cu",
                    "src/kernels/cuda/flash_attention_cpu.cpp",
                ],
                extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"] + arch_flags},
            )
        )

        # Paged attention extension
        extensions.append(
            CUDAExtension(
                name="mini_yaie_kernels.paged_attention",
                sources=[
                    "src/kernels/cuda/paged_attention.cu",
                    "src/kernels/cuda/paged_attention_cpu.cpp",
                ],
                extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"] + arch_flags},
            )
        )

        # Radix attention extension
        extensions.append(
            CUDAExtension(
                name="mini_yaie_kernels.radix_attention",
                sources=[
                    "src/kernels/cuda/radix_attention.cu",
                    "src/kernels/cuda/radix_attention_cpu.cpp",
                ],
                extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"] + arch_flags},
            )
        )

        # Memory management extension
        extensions.append(
            CUDAExtension(
                name="mini_yaie_kernels.memory_ops",
                sources=[
                    "src/kernels/cuda/memory_ops.cu",
                    "src/kernels/cuda/memory_ops_cpu.cpp",
                ],
                extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"] + arch_flags},
            )
        )

    return extensions


if __name__ == "__main__":
    print("Building YAIE CUDA kernels...")

    # Create directories if they don't exist
    os.makedirs("src/kernels/cuda", exist_ok=True)

    extensions = build_extensions()

    if extensions:
        setup(
            name="mini_yaie_kernels",
            ext_modules=extensions,
            cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
            packages=[
                "mini_yaie_kernels"
            ],  # This is just a placeholder - actual package will be built
            packages=['mini_yaie_kernels'],  # This is just a placeholder - actual package will be built
        )
        print("CUDA kernels built successfully!")
    else:
        print("No CUDA kernels built (CUDA not available)")
        print("The engine will run in CPU-only mode")
