# Radix Attention Mechanism (`radix_attention.py` and `radix_tree.py`)

This section documents the implementation of the Radix Attention mechanism, a key component inspired by SGLang for efficient Large Language Model (LLM) inference. It encompasses both the attention calculation itself (`radix_attention.py`) and the underlying data structure for prefix sharing (`radix_tree.py`).

## `radix_attention.py`

This module contains the core attention block and a paged KV-cache management system tailored for radial attention.

### Helper Functions

*   `rotate_half(x)`:
    Rotates half of the hidden dimensions of the input tensor. This is a common operation used in Rotary Positional Embeddings (RoPE).

*   `apply_rotary_pos_emb(q, k, cos, sin, position_ids=None)`:
    Applies Rotary Positional Embeddings (RoPE) to the query (`q`) and key (`k`) tensors. RoPE helps in encoding absolute position information with a decaying effect on long-range dependencies.

### `RadixAttentionBlock` Class

Implements a single block of self-attention with prefix sharing capabilities, incorporating Rotary Positional Embeddings.

*   **Purpose**: Performs the attention computation for a given layer, taking into account previous key-value states and applying RoPE.

*   **`__init__(self, hidden_size: int, num_heads: int, head_dim: int, max_position_embeddings: int = 2048, rope_theta: float = 10000.0)`**:
    Initializes the attention block.
    *   **Parameters**:
        *   `hidden_size` (int): The dimensionality of the input and output features.
        *   `num_heads` (int): The number of attention heads.
        *   `head_dim` (int): The dimensionality of each attention head.
        *   `max_position_embeddings` (int): The maximum sequence length for positional embeddings.
        *   `rope_theta` (float): A hyperparameter for the Rotary Positional Embeddings.
    *   **Initialization**: Sets up linear projections for Query, Key, Value, and Output. Pre-computes and caches sinusoidal positional embeddings (`cos_cached`, `sin_cached`) for RoPE.

*   **`_setup_rope_embeddings(self)`**:
    A private method to initialize the Rotary Positional Embeddings based on `max_position_embeddings` and `rope_theta`.

*   **`forward(...)`**:
    Performs the forward pass of the attention block.
    *   **Input Parameters**:
        *   `hidden_states` (torch.Tensor): Input tensor of shape `[batch_size, seq_len, hidden_size]`.
        *   `attention_mask` (Optional[torch.Tensor]): Mask to prevent attention to padding tokens or future tokens in causal models.
        *   `position_ids` (Optional[torch.Tensor]): Positional indices for applying RoPE.
        *   `past_key_value` (Optional[Tuple[torch.Tensor, torch.Tensor]]): Tuple of previous key and value states for incremental decoding.
        *   `output_attentions` (bool): If `True`, attention weights are returned.
        *   `use_cache` (bool): If `True`, the current key-value states are returned for caching.
    *   **Process**:
        1.  Projects `hidden_states` into Query, Key, and Value tensors.
        2.  Reshapes and transposes QKV for multi-head attention.
        3.  Applies Rotary Positional Embeddings to Query and Key.
        4.  Concatenates `past_key_value` if provided (for incremental decoding).
        5.  Computes scaled dot-product attention scores (`Q @ K.T / sqrt(head_dim)`).
        6.  Applies `attention_mask` to scores.
        7.  Applies softmax to get attention probabilities.
        8.  Multiplies probabilities with Value tensor to get attention output.
        9.  Reshapes and projects the output back to `hidden_size`.
    *   **Returns**: A tuple containing the output tensor, optional attention weights, and optional updated key-value pairs.

### `RadixAttentionWithPagedKVCache` Class

Manages a paged Key-Value cache specifically for use with the radial attention mechanism. This class handles the allocation and management of KV-cache blocks for individual requests.

*   **Purpose**: Provides an interface for attention mechanisms to store and retrieve KV states efficiently across different requests, supporting continuous batching and prefix sharing.

*   **`__init__(self, num_layers: int, num_heads: int, head_dim: int, block_size: int = 16, max_blocks_per_request: int = 128)`**:
    Initializes the paged KV-cache manager for radial attention.
    *   **Parameters**:
        *   `num_layers` (int): Number of transformer layers.
        *   `num_heads` (int): Number of attention heads.
        *   `head_dim` (int): Dimension of each attention head.
        *   `block_size` (int): Number of tokens per KV-cache block.
        *   `max_blocks_per_request` (int): Maximum blocks a single request can use.
    *   **Initialization**: Sets up empty dictionaries for `key_cache_pool`, `value_cache_pool`, and `request_block_map`. Manages `next_block_index` and `max_total_blocks`.

*   **`allocate_blocks(self, request_id: str, num_blocks_needed: int) -> List[int]`**:
    Allocates new KV-cache blocks from the pool for a given request.
    *   **Parameters**:
        *   `request_id` (str): Unique identifier for the request.
        *   `num_blocks_needed` (int): The number of blocks to allocate.
    *   **Returns**: A list of `block_id`s allocated to the request.
    *   **Raises**: `RuntimeError` if there's insufficient space (in a production system, this would trigger an eviction policy).

*   **`append_slot(self, key: torch.Tensor, value: torch.Tensor, request_id: str)`**:
    Appends new key-value pairs to the cache for a specific request. This method handles filling existing blocks and allocating new ones if the current block is full.

*   **`get_kv_cache(self, request_ids: List[str], seq_lens: List[int]) -> Tuple[torch.Tensor, torch.Tensor]`**:
    Retrieves and concatenates the key and value tensors for a list of specified requests, respecting their sequence lengths.

*   **`free_request(self, request_id: str)`**:
    Frees the KV-cache blocks associated with a given request. In this simplified implementation, it removes references, but a full system would manage actual memory deallocation and reuse.

## `radix_tree.py`

This module implements a Radix Tree data structure, a crucial component for enabling SGLang-style prefix matching and sharing across multiple concurrent requests. By identifying common prefixes, it allows for sharing of computation, reducing redundant work.

### `RadixTreeNode` Class

Represents a single node within the Radix Tree.

*   **Purpose**: Stores information about a token, its children in the tree, and the requests that traverse this node.

*   **`__init__(self, token_id: Optional[int] = None)`**:
    Initializes a tree node.
    *   **Attributes**:
        *   `token_id` (Optional[int]): The token ID represented by this node (None for the root).
        *   `children` (Dict[int, "RadixTreeNode"]): A dictionary mapping child token IDs to their respective `RadixTreeNode` objects.
        *   `request_ids` (List[str]): A list of unique request identifiers that have traversed this node's prefix path.
        *   `kv_cache_refs` (List[str]): Placeholder for references to KV cache blocks (currently not fully utilized in this version).
        *   `is_terminal` (bool): Indicates if this node marks the end of a complete request's prefix.

### `RadixTree` Class

Manages the overall Radix Tree structure, allowing for insertion, removal, and querying of request prefixes.

*   **Purpose**: Efficiently groups requests with common prefixes to facilitate shared computation.

*   **`__init__(self)`**:
    Initializes the Radix Tree with a root node and internal mappings.
    *   **Attributes**:
        *   `root` (RadixTreeNode): The root node of the tree.
        *   `request_to_path` (Dict[str, List[int]]): Maps `request_id` to its token path.
        *   `path_to_node` (Dict[str, RadixTreeNode]): Maps a string representation of a token path to its corresponding `RadixTreeNode`.

*   **`insert_request(self, request_id: str, token_ids: List[int])`**:
    Inserts a request's token sequence (prompt) into the Radix Tree, creating new nodes as necessary and updating `request_ids` along the path.

*   **`find_shared_prefixes(self, token_ids: List[int]) -> Tuple[List[str], int]`**:
    Traverses the tree with a given token sequence to identify all requests that share a common prefix with it.
    *   **Returns**: A tuple containing a list of unique `request_id`s that share a prefix and the length of the longest shared prefix.

*   **`remove_request(self, request_id: str)`**:
    Removes a request from the Radix Tree. It cleans up the `request_ids` from nodes along the request's path and prunes any nodes that become unused (no associated requests and no children).

*   **`_cleanup_unused_nodes(self, token_path: List[int])`**:
    A private helper method that performs a bottom-up cleanup of the tree, removing nodes that are no longer part of any active request and have no children.

*   **`get_shared_computation_graph(self) -> Dict[str, Any]`**:
    Analyzes the tree structure to identify and represent opportunities for shared computation.
    *   **Returns**: A dictionary detailing nodes, edges, sharing opportunities, and potential savings.

*   **`_analyze_sharing_opportunities(self, graph_dict: Dict[str, Any])`**:
    A private helper to traverse the tree and identify specific nodes where multiple requests converge, indicating a sharing opportunity.

*   **`_traverse_for_sharing(self, node: RadixTreeNode) -> Dict[str, Any]`**:
    A private recursive helper for traversing the tree and building a structural representation of sharing.

*   **`_path_to_string(self, token_ids: List[int]) -> str`**:
    Converts a list of token IDs into a string key for internal mapping.

*   **`get_prefix_groups(self) -> Dict[str, List[str]]`**:
    Identifies and groups requests that share common prefixes.
    *   **Returns**: A dictionary where keys are string representations of shared prefixes and values are lists of `request_id`s belonging to that group.

*   **`_collect_groups(self, node: RadixTreeNode, groups: Dict, current_path: List[int])`**:
    A private recursive helper to traverse the tree and collect prefix groups.

### `RequestPrefixMatcher` Class

A higher-level system that utilizes the `RadixTree` for SGLang-style prefix matching to group requests and prepare for computation sharing.

*   **Purpose**: Simplifies the process of adding requests, finding sharing opportunities, and analyzing optimization potential.

*   **`__init__(self)`**:
    Initializes the matcher with a `RadixTree` instance.

*   **`add_request_prefix(self, request_id: str, prompt_tokens: List[int])`**:
    Adds a request's tokenized prompt to the underlying `RadixTree` and updates internal prefix similarity measures.

*   **`_update_prefix_similarity(self, request_id: str, prompt_tokens: List[int])`**:
    A private method to update similarity metrics based on shared prefixes. (Currently a simplified placeholder).

*   **`find_computation_sharing_opportunities(self, new_request_id: str, new_tokens: List[int]) -> List[str]`**:
    Queries the `RadixTree` to find existing requests whose prefixes match that of a new request, indicating potential for shared computation.

*   **`remove_request(self, request_id: str)`**:
    Removes a request from the prefix matching system by delegating to the `RadixTree`'s removal method.

*   **`get_optimization_suggestions(self) -> Dict[str, Any]`**:
    Analyzes the current state of the Radix Tree to provide suggestions for computation optimization, including shared graph details, prefix groups, and estimated speedup.

*   **`_estimate_speedup(self, prefix_groups: Dict[str, List[str]]) -> float`**:
    A private helper method to estimate the potential speedup achievable through prefix sharing based on the identified prefix groups.
