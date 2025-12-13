# Troubleshooting

## "CUDA kernel not found"

- Ensure you ran `pip install -e .`.
- Check if `nvcc` is in your path: `nvcc --version`.

## "OutOfMemoryError"

- Decrease `max_batch_size`.
- Decrease `kv_cache_manager` block count.

## "ImportError: attempted relative import..."

- Ensure you are running the `yaie` command, or running python as a module `python -m src.cli.main`.
- Do not run scripts directly like `python src/engine.py`.
