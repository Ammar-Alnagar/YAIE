#!/bin/bash

# Build script for Mini-YAIE custom kernels
set -e  # Exit on any error

echo "Building Mini-YAIE custom kernels..."

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "CUDA not found. Building CPU-only version."
    echo "For GPU acceleration, please install CUDA toolkit."
    exit 0
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Python not found. Please install Python 3.8+."
    exit 1
fi

# Check for PyTorch with CUDA support
python -c "import torch; assert torch.cuda.is_available(), 'PyTorch CUDA not available'" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "PyTorch with CUDA support not found. Please install PyTorch with CUDA support."
    exit 1
fi

echo "Found CUDA and PyTorch with CUDA support. Starting build..."

# Create build directory if it doesn't exist
mkdir -p build

# Build the extensions
python setup_kernels.py build_ext --inplace

echo "Build completed successfully!"
echo "To verify the installation:"
echo "  python -c \"import torch; from src.kernels import *; print('Kernels loaded successfully')\""
