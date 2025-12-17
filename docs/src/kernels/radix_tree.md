# Radix Tree: SGLang-Style Prefix Matching System

## Overview

The Radix Tree system in Mini-YAIE implements SGLang-style prefix matching and request grouping for computational optimization. It enables efficient identification of requests with common prefixes, allowing the system to share computation and maximize throughput by processing shared prefixes only once.

## Core Concepts

### Radix Trees vs Traditional Data Structures

Unlike binary search trees or hash tables, radix trees (also known as tries) organize data by the actual characters/elements of the keys. For LLM inference, this means organizing request prefixes by their token sequences.

**Benefits for LLM Inference:**
- Efficient prefix matching for request grouping
- Fast retrieval of shared computation opportunities
- Hierarchical structure matching the nature of text sequences

### SGLang-Style Prefix Matching

The implementation follows SGLang principles by:
- Identifying requests with common prefixes early
- Enabling shared computation for matched prefixes
- Optimizing batch formation based on prefix similarity
- Maximizing computational efficiency through reuse

## Architecture

### RadixTreeNode Structure

The core data structure is the `RadixTreeNode`:

```python
class RadixTreeNode:
    def __init__(self, token_id: Optional[int] = None):
        self.token_id = token_id              # Token ID for this node (None for root)
        self.children: Dict[int, "RadixTreeNode"] = {}  # Child nodes
        self.request_ids: List[str] = []      # Requests passing through this node
        self.kv_cache_refs: List[str] = []    # References to KV cache blocks
        self.is_terminal = False              # Whether this marks the end of a prefix
```

#### Node Properties

- **token_id**: The token that leads to this node (None for the root)
- **children**: Dictionary mapping token IDs to child nodes in the tree
- **request_ids**: List of request IDs that pass through this node (for prefix sharing)
- **kv_cache_refs**: References to associated KV-cache blocks
- **is_terminal**: Indicates if this node represents the end of a complete prefix

### RadixTree Class

The main `RadixTree` class manages the tree structure and operations:

```python
class RadixTree:
    def __init__(self):
        self.root = RadixTreeNode()
        self.request_to_path: Dict[str, List[int]] = {}  # Maps requests to token paths
        self.path_to_node: Dict[str, RadixTreeNode] = {}  # Maps paths to nodes
```

#### Core Data Structures

- **root**: The root node of the tree with no token_id
- **request_to_path**: Maps request ID to its token sequence for quick lookup
- **path_to_node**: Maps path strings to corresponding nodes for efficient access

## Key Operations

### 1. Request Insertion

The `insert_request` method adds a request to the radix tree:

```python
def insert_request(self, request_id: str, token_ids: List[int]):
    current = self.root

    # Navigate/create path for each token
    for token_id in token_ids:
        if token_id not in current.children:
            current.children[token_id] = RadixTreeNode(token_id)
        current = current.children[token_id]
        # Add request to all nodes along the path for prefix matching
        if request_id not in current.request_ids:
            current.request_ids.append(request_id)

    current.is_terminal = True
    self.request_to_path[request_id] = token_ids
    path_str = self._path_to_string(token_ids)
    self.path_to_node[path_str] = current
```

#### Insertion Process

1. Start at the root node
2. For each token in the sequence:
   - If the token doesn't exist as a child, create a new node
   - Move to the child node
   - Add the request ID to the node's request list
3. Mark the final node as terminal
4. Update internal mappings for fast access

### 2. Prefix Matching

The `find_shared_prefixes` method identifies requests that share common prefixes:

```python
def find_shared_prefixes(self, token_ids: List[int]) -> Tuple[List[str], List[int]]:
    current = self.root
    matched_requests = []
    prefix_length = 0

    for i, token_id in enumerate(token_ids):
        if token_id in current.children:
            current = current.children[token_id]
            matched_requests.extend(current.request_ids)
            prefix_length = i + 1
        else:
            break  # No more matching
    
    return list(set(matched_requests)), prefix_length
```

#### Matching Process

1. Traverse the tree with the given token sequence
2. At each matching node, collect associated request IDs
3. Track the length of the matched prefix
4. Stop when no matching child node exists
5. Return unique matching request IDs and prefix length

### 3. Request Removal

The `remove_request` method removes a request from the tree:

```python
def remove_request(self, request_id: str):
    if request_id not in self.request_to_path:
        return

    token_path = self.request_to_path[request_id]
    current = self.root

    # Remove from each node along the path
    for token_id in token_path:
        if token_id in current.children:
            current = current.children[token_id]
            if request_id in current.request_ids:
                current.request_ids.remove(request_id)

    # Clean up mappings
    del self.request_to_path[request_id]
    path_str = self._path_to_string(token_path)
    if path_str in self.path_to_node:
        del self.path_to_node[path_str]
```

## SGLang-Style Optimization Features

### 1. Computation Graph Analysis

The radix tree can analyze shared computation opportunities:

```python
def get_shared_computation_graph(self) -> Dict[str, Any]:
    return self._traverse_for_sharing(self.root)

def _traverse_for_sharing(self, node: RadixTreeNode) -> Dict[str, Any]:
    result = {
        "token_id": node.token_id,
        "request_count": len(node.request_ids),
        "children": {},
        "is_shared": len(node.request_ids) > 1,  # True if multiple requests share this
    }

    for token_id, child_node in node.children.items():
        result["children"][token_id] = self._traverse_for_sharing(child_node)

    return result
```

This provides insight into:
- Number of requests sharing each prefix
- Shared computation opportunities
- Potential optimization benefits

### 2. Prefix Grouping

The tree can identify groups of requests with shared prefixes:

```python
def get_prefix_groups(self) -> Dict[str, List[str]]:
    groups = {}
    self._collect_groups(self.root, groups, [])
    return groups

def _collect_groups(self, node: RadixTreeNode, groups: Dict, current_path: List[int]):
    if len(node.request_ids) > 1:  # Multiple requests share this prefix
        path_str = self._path_to_string(current_path)
        groups[path_str] = node.request_ids[:]

    for token_id, child_node in node.children.items():
        self._collect_groups(child_node, groups, current_path + [token_id])
```

## RequestPrefixMatcher Class

The `RequestPrefixMatcher` provides a high-level interface for prefix matching:

```python
class RequestPrefixMatcher:
    def __init__(self):
        self.radix_tree = RadixTree()
        self.max_prefix_cache_size = 10000
```

### Core Operations

#### Adding Request Prefixes

```python
def add_request_prefix(self, request_id: str, prompt_tokens: List[int]):
    self.radix_tree.insert_request(request_id, prompt_tokens)
```

#### Finding Sharing Opportunities

```python
def find_computation_sharing_opportunities(
    self, new_request_id: str, new_tokens: List[int]
) -> List[str]:
    shared_requests, _ = self.radix_tree.find_shared_prefixes(new_tokens)
    return shared_requests
```

#### Optimization Analysis

```python
def get_optimization_suggestions(self) -> Dict[str, Any]:
    sharing_graph = self.radix_tree.get_shared_computation_graph()
    prefix_groups = self.radix_tree.get_prefix_groups()

    return {
        "sharing_graph": sharing_graph,
        "prefix_groups": prefix_groups,
        "total_sharing_opportunities": len(prefix_groups),
        "estimated_speedup_factor": self._estimate_speedup(prefix_groups),
    }
```

## Performance Considerations

### Time Complexity

- **Insertion**: O(|T|) where |T| is the length of the token sequence
- **Search**: O(|S|) where |S| is the length of the search sequence
- **Deletion**: O(|T|) where |T| is the length of the token sequence

### Space Complexity

- O(ALPHABET_SIZE × N × M) in the worst case
- Where N is the number of requests and M is the average prefix length
- Much more efficient in practice due to shared prefixes

### Memory Optimization Strategies

- Token compression for common sequences
- Node sharing for identical suffixes
- Lazy evaluation for deep paths

## Integration with Inference Engine

### Scheduler Integration

The radix tree connects with the SGLang scheduler to:
- Group requests with common prefixes
- Optimize batch formation
- Maximize computational sharing opportunities

### Memory Management

Works with the paged KV-cache system to:
- Track shared cache blocks
- Optimize memory allocation for shared prefixes
- Manage cache references efficiently

### Attention Mechanism

Provides information to the radial attention system about:
- Which requests share computation
- How to organize shared computations
- When requests diverge and require separate processing

## Advanced Features

### 1. Dynamic Resizing

The system could implement:
- Tree balancing for performance
- Node merging for common suffixes
- Memory-efficient storage of sparse trees

### 2. Approximate Matching

Future implementations could include:
- Fuzzy prefix matching for similar requests
- Semantic similarity beyond exact token matching
- Adaptive matching thresholds

### 3. Cache-Aware Optimizations

Advanced features might include:
- Prefetching based on tree traversal patterns
- Cache-aware node placement
- Hierarchical caching strategies

## Implementation Challenges

### 1. Memory Management

- Efficient tracking of node references
- Garbage collection for unused branches
- Memory sharing across requests

### 2. Concurrency

- Thread-safe operations in multi-request scenarios
- Lock-free implementations for performance
- Consistency across tree operations

### 3. Scalability

- Performance with large numbers of requests
- Efficient handling of very long prefixes
- Memory usage optimization for production scale

## SGLang-Style Optimization Benefits

### 1. Throughput Improvement

By sharing computation for common prefixes:
- Reduces redundant calculations
- Increases effective batch utilization
- Maximizes GPU efficiency

### 2. Memory Efficiency

- Shared KV-cache blocks for common prefixes
- Reduced overall memory requirements
- Better memory utilization patterns

### 3. Latency Reduction

- Faster processing for requests with common starts
- More efficient cache usage
- Reduced end-to-end response times

## Future Enhancements

### 1. Advanced Tree Algorithms

- Suffix tree integration for even more optimization
- Compressed representations for memory efficiency
- Adaptive structures based on request patterns

### 2. Machine Learning Integration

- Predictive prefix matching based on historical patterns
- Learned optimization strategies
- Dynamic threshold adjustment

### 3. Distributed Extensions

- Distributed radix trees for multi-GPU setups
- Consistent hashing for load balancing
- Cross-node prefix matching

This radix tree system forms the foundation for SGLang-style optimization in Mini-YAIE, enabling efficient prefix matching and computation sharing that dramatically improves LLM inference performance.