# Environment Setup

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Mini-YAIE.git
cd Mini-YAIE
```

## 2. Python Environment

It is highly recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 3. CUDA Requirements (Optional)

To build and run the CUDA kernels, you need:

- NVIDIA GPU (Compute Capability 7.0+)
- CUDA Toolkit 11.8+
- PyTorch with CUDA support

If you do not have a GPU, you can still implement the Python logic and the CPU fallback kernels.

## 4. Documentation Setup

To serve this documentation locally:

1.  **Install mdbook**:

    ```bash
    # If you have Rust/Cargo installed:
    cargo install mdbook

    # Or download the binary from their GitHub releases.
    ```

2.  **Serve the docs**:
    ```bash
    mdbook serve docs
    ```
    Navigate to `http://localhost:3000` in your browser.
