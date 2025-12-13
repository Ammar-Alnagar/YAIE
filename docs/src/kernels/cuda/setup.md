# CUDA Setup

Implementing custom CUDA kernels requires compiling C++/CUDA code and binding it to Python.

## `setup.py`

We use `torch.utils.cpp_extension` to handle compilation.
The `setup.py` file in the root directory is already configured to look for kernels in `src/kernels/cuda/`.

### Triggering Compilation

To compile your kernels, simply run:

```bash
pip install -e .
```

This command invokes `nvcc` (NVIDIA CUDA Compiler) on your `.cu` files.

### Using the Kernels

Once compiled, you can import them in Python:

```python
import mini_yaie_kernels

# Call your C++ function
mini_yaie_kernels.flash_attention.forward(...)
```
