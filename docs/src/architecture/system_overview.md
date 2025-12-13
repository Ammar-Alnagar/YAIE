# System Overview

Mini-YAIE follows a modular architecture similar to vLLM.

## High-Level Components

```mermaid
graph TD
    User[User / API Client] --> API[FastAPI Server (server/api.py)]
    API --> Engine[Inference Engine (engine.py)]

    subgraph Core Logic
        Engine --> Scheduler[SGLang Scheduler (core/sglang_scheduler.py)]
        Engine --> MM[Memory Manager (kernels/kv_cache.py)]
        Engine --> Model[LLM Model (HuggingFace)]
    end

    subgraph Kernels
        Scheduler --> RadixTree[Radix Tree (kernels/radix_tree.py)]
        Model --> RadixAttn[Radix Attention (kernels/radix_attention.py)]
        Model --> PagedAttn[Paged Attention (kernels/cuda/paged_attention.cu)]
    end
```

## Data Flow

1.  **Request**: User sends a prompt to the API.
2.  **Scheduling**: Scheduler analyzes the prompt, checks the Radix Tree for cached prefixes, and assigns a Request ID.
3.  **Batching**: `schedule_step` groups requests into Prefill (new) and Decode (running) batches.
4.  **Execution**: The Engine runs the model.
    - **Prefill**: Computes initial KV cache for new prompts.
    - **Decode**: Generates one token for running requests.
5.  **Memory**: The Memory Manager allocates/frees GPU blocks as needed.
