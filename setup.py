import os
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


def get_extensions():
    """Conditionally create CUDA extensions if CUDA is available and files exist."""
    extensions = []

    # Check if we have the CUDA kernel files
    cuda_dir = Path("src/kernels/cuda")
    kernel_files_exist = (
        (cuda_dir / "flash_attention.cu").exists() and
        (cuda_dir / "flash_attention_cpu.cpp").exists()
    )

    if not kernel_files_exist:
        print("CUDA kernel files not found, skipping CUDA extensions")
        return extensions

    try:
        import torch
        if torch.cuda.is_available():
            arch_flags = []
            # Get the compute capability of the first available GPU
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                arch_flag = f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
                arch_flags = [arch_flag]

            extensions.extend([
                CUDAExtension(
                    name="mini_yaie_kernels.flash_attention",
                    sources=[
                        "src/kernels/cuda/flash_attention.cu",
                        "src/kernels/cuda/flash_attention_cpu.cpp",
                    ],
                    extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"] + arch_flags},
                ),
                CUDAExtension(
                    name="mini_yaie_kernels.paged_attention",
                    sources=[
                        "src/kernels/cuda/paged_attention.cu",
                        "src/kernels/cuda/paged_attention_cpu.cpp",
                    ],
                    extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"] + arch_flags},
                ),
                CUDAExtension(
                    name="mini_yaie_kernels.radix_attention",
                    sources=[
                        "src/kernels/cuda/radix_attention.cu",
                        "src/kernels/cuda/radix_attention_cpu.cpp",
                    ],
                    extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"] + arch_flags},
                ),
                CUDAExtension(
                    name="mini_yaie_kernels.memory_ops",
                    sources=[
                        "src/kernels/cuda/memory_ops.cu",
                        "src/kernels/cuda/memory_ops_cpu.cpp",
                    ],
                    extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"] + arch_flags},
                ),
            ])
        else:
            print("CUDA not available, skipping CUDA extensions")
    except ImportError:
        print("PyTorch not available, skipping CUDA extensions")

    return extensions


def get_package_data():
    """Get package data including CUDA files if they exist."""
    package_data = []
    cuda_dir = Path("src/kernels/cuda")

    # Include CUDA related files if they exist
    cuda_files = ["*.cu", "*.cpp", "*.h", "*.hpp"]
    for pattern in cuda_files:
        import glob
        files = glob.glob(f"src/kernels/cuda/{pattern}")
        if files:
            package_data.extend([f"kernels/cuda/{pattern}"])

    return {"src": package_data}


# Get extensions
extensions = get_extensions()

# Setup the package
setup(
    name="mini-yaie",
    version="0.1.0",
    description="Educational LLM Inference Engine with Radix Attention and FlashInfer concepts",
    author="Ammar",
    author_email="ammar@example.com",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data=get_package_data(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.0.0",
        "tokenizers>=0.13.0",
        "huggingface-hub>=0.14.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "click>=8.0.0",
        "pydantic>=2.0.0",
        "numpy>=1.21.0",
        "sentencepiece>=0.1.96",
        "accelerate>=0.20.0",
        "peft>=0.4.0",
        "bitsandbytes>=0.39.0",
        "scipy>=1.10.0",
        "pynvml>=11.4.1"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "yaie = cli.main:main",
        ],
    },
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExtension} if extensions else {},
)