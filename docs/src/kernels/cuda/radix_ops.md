# Radix Operations (`kernels/cuda/radix_ops.cu`)

## Concept

If we have a Radix Tree, we can optimize attention even further by knowing exactly which tokens are shared.

## Implementation Goal

This is an advanced extension.

### Logic

1.  **Tree Traversal on GPU**: Mapping the Radix Tree structure to a GPU-friendly format (e.g., flattened arrays).
2.  **Prefix Matching**: A kernel that takes a batch of prompts and quickly identifies the longest common prefix node ID for each.

_Note: In the simplified version, this logic is often kept in CPU (Python) and only the KV indices are passed to the GPU._
