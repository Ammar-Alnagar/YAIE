# CLI Usage

## Starting the Server

```bash
yaie serve --model gpt2 --port 8000
```

## Interactive Chat

```bash
yaie chat --model gpt2
```

## Arguments

- `--model`: Name of the HuggingFace model or path to local model.
- `--max-batch-size`: Limit the number of concurrent requests.
- `--gpu-memory-utilization`: Fraction of GPU memory to use for KV cache (default 0.9).
