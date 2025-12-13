# Production Considerations

While Mini-YAIE is educational, here is what you would need for production:

## 1. Batching & Latency

- **Timeouts**: Requests waiting too long in the queue should be handled.
- **Preemption**: If high-priority requests come in, lower priority ones should be paused (swapped to CPU).

## 2. Distributed Inference

- **Tensor Parallelism**: Splitting the model weights across multiple GPUs (Megatron-LM style).
- **Pipeline Parallelism**: Splitting layers across GPUs.

## 3. Quantization

- **FP8 / INT8**: Running with lower precision to save memory and increase compute speed (using library like `bitsandbytes`).
