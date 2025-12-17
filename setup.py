import os
from pathlib import Path
from setuptools import setup, find_packages


def get_package_data():
    """Get package data including CUDA files if they exist."""
    package_data = []
    cuda_dir = Path("src/kernels/cuda")

    # Include CUDA related files if they exist (for distribution)
    cuda_files = ["*.cu", "*.cpp", "*.h", "*.hpp"]
    for pattern in cuda_files:
        import glob
        files = glob.glob(f"src/kernels/cuda/{pattern}")
        if files:
            package_data.extend([f"kernels/cuda/{pattern}"])

    return {"src": package_data}


# Setup the package without CUDA extensions for now to avoid install issues
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
    # No CUDA extensions to avoid build issues with CUDA version mismatch
)
