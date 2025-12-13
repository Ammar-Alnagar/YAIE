# Radix Tree (`kernels/radix_tree.py`)

## Concept

A **Radix Tree** (or Compressed Trie) is a space-optimized prefix tree.

## Implementation Goal

You need to implement the `RadixTree` class with:

### 1. `insert(token_ids, request_id)`

- Traverse the tree with `token_ids`.
- If a path exists, follow it.
- If tokens diverge, **split** the edge and create a new node.
- Store the `request_id` at the leaf.

### 2. `match_prefix(token_ids)`

- Traverse the tree to find the longest common prefix.
- Return the `node_id` and the length of the match.
- This tells the scheduler how many tokens we can skip computing!

### 3. `remove(request_id)`

- When a request finishes, decrement reference counts.
- If a node has no references, free its associated KV cache blocks.
