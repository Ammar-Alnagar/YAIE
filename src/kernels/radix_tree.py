"""
Radix Tree implementation for SGLang-style prefix matching
Enables efficient identification of shared prefixes for computation sharing
"""

import hashlib
from typing import Any, Dict, List, Optional, Tuple


class RadixTreeNode:
    """Node in the Radix Tree for prefix matching"""

    def __init__(self, token_id: Optional[int] = None):
        self.token_id = token_id  # Token ID for this node (None for root)
        self.children: Dict[int, "RadixTreeNode"] = {}  # Child nodes
        self.request_ids: List[str] = []  # Requests that pass through this node
        self.kv_cache_refs: List[str] = []  # References to KV cache blocks
        self.is_terminal = False  # Whether this is the end of a prefix


class RadixTree:
    """
    Radix Tree for SGLang-style prefix matching and sharing
    Efficiently groups requests with common prefixes for shared computation
    """

    def __init__(self):
        self.root = RadixTreeNode()
        self.request_to_path: Dict[
            str, List[int]
        ] = {}  # Maps requests to their token paths
        self.path_to_node: Dict[str, RadixTreeNode] = {}  # Maps paths to nodes

    def insert_request(self, request_id: str, token_ids: List[int]):
        """
        Insert a request into the radix tree based on its token sequence

        Args:
            request_id: ID of the request
            token_ids: Token IDs representing the prompt
        """
        current = self.root

        # For each token in the sequence, navigate/create the path
        # Add the request ID to each node along the path
        for token_id in token_ids:
            if token_id not in current.children:
                current.children[token_id] = RadixTreeNode(token_id)
            current = current.children[token_id]
            # Add request to this node along the path (for prefix matching)
            if request_id not in current.request_ids:
                current.request_ids.append(request_id)

        current.is_terminal = True

        # Store mapping from request to its path
        self.request_to_path[request_id] = token_ids

        # Create path string for node mapping
        path_str = self._path_to_string(token_ids)
        self.path_to_node[path_str] = current

    def find_shared_prefixes(self, token_ids: List[int]) -> Tuple[List[str], int]:
        """
        Find requests that share prefixes with the given token sequence

        Args:
            token_ids: Token sequence to match against

        Returns:
            Tuple of (request_ids_with_shared_prefix, shared_prefix_length)
        """
        # Implement efficient prefix matching for SGLang optimization:
        # 1. Traverse tree with given token sequence
        # 2. Identify all nodes that match (shared prefixes)
        # 3. Collect all requests that share those prefixes
        # 4. Return shared computation opportunities
        current = self.root
        matched_requests = []
        prefix_length = 0

        # Track all requests at each level of the prefix
        all_matched_requests = []

        for i, token_id in enumerate(token_ids):
            if token_id in current.children:
                current = current.children[token_id]
                # Add all requests that have this prefix
                all_matched_requests.extend(current.request_ids)
                prefix_length = i + 1
            else:
                # No more matching, but we can still use previously matched requests
                break

        # Return unique request IDs and the length of shared prefix
        unique_requests = list(set(all_matched_requests))
        return unique_requests, prefix_length

    def remove_request(self, request_id: str):
        """
        Remove a request from the radix tree

        Args:
            request_id: ID of request to remove
        """
        # Implement radix tree removal:
        # 1. Find the path for the request
        # 2. Remove request from all nodes along the path
        # 3. Clean up unused nodes (garbage collection)
        # 4. Update mappings accordingly
        if request_id not in self.request_to_path:
            return

        token_path = self.request_to_path[request_id]
        current = self.root

        # Store the path to check for cleanup later
        path_nodes = [current]  # Include root node
        path_keys = []

        # Remove from each node along the path
        for token_id in token_path:
            if token_id in current.children:
                current = current.children[token_id]
                path_nodes.append(current)
                path_keys.append(token_id)

                if request_id in current.request_ids:
                    current.request_ids.remove(request_id)

        # Clean up the mappings
        del self.request_to_path[request_id]
        path_str = self._path_to_string(token_path)
        if path_str in self.path_to_node:
            del self.path_to_node[path_str]

        # Perform cleanup of unused nodes (bottom-up)
        self._cleanup_unused_nodes(token_path)

    def _cleanup_unused_nodes(self, token_path: List[int]):
        """Remove nodes that no longer have any associated requests and have no children."""
        if not token_path:
            return

        # Navigate to the terminal node and work backwards
        current = self.root
        path_nodes = [self.root]

        for token_id in token_path:
            current = current.children.get(token_id)
            if current:
                path_nodes.append(current)

        # Clean up bottom-up
        for i in range(len(path_nodes) - 1, 0, -1):  # Skip root node (index 0)
            node = path_nodes[i]
            parent = path_nodes[i-1]

            # Find the key to this node
            node_key = token_path[i-1] if i-1 < len(token_path) else None

            # If node has no requests and no children, remove it
            if len(node.request_ids) == 0 and len(node.children) == 0:
                if node_key is not None and node_key in parent.children:
                    del parent.children[node_key]

    def get_shared_computation_graph(self) -> Dict[str, Any]:
        """
        Get representation of shared computation opportunities

        Returns:
            Dict representing the computation sharing structure
        """
        # Implement computation graph analysis for SGLang optimization:
        # 1. Analyze the tree to identify shared computation paths
        # 2. Calculate potential savings from sharing
        # 3. Provide information for scheduler optimization
        computation_graph = {
            "nodes": [],
            "edges": [],
            "sharing_opportunities": [],
            "potential_savings": 0,
            "structure": self._traverse_for_sharing(self.root),
        }

        # Analyze and add computation sharing opportunities
        self._analyze_sharing_opportunities(computation_graph)

        return computation_graph

    def _analyze_sharing_opportunities(self, graph_dict: Dict[str, Any]):
        """Analyze the tree to find specific sharing opportunities."""
        sharing_opps = []

        # Walk through the tree to identify nodes with multiple associated requests
        def walk_tree(node, path=[]):
            if len(node.request_ids) > 1:
                # This is a sharing opportunity
                sharing_opps.append({
                    "path": path.copy(),
                    "token_id": node.token_id,
                    "request_count": len(node.request_ids),
                    "requests": node.request_ids.copy(),
                    "potential_savings": len(node.request_ids) - 1,  # N computations saved to 1
                })

            for token_id, child_node in node.children.items():
                walk_tree(child_node, path + [token_id])

        walk_tree(self.root)

        graph_dict["sharing_opportunities"] = sharing_opps
        # Calculate aggregate potential savings
        total_savings = sum(opp["potential_savings"] for opp in sharing_opps)
        graph_dict["potential_savings"] = total_savings

    def _traverse_for_sharing(self, node: RadixTreeNode) -> Dict[str, Any]:
        """Helper to traverse and identify sharing opportunities"""
        result = {
            "token_id": node.token_id,
            "request_count": len(node.request_ids),
            "children": {},
            "is_shared": len(node.request_ids) > 1,
        }

        for token_id, child_node in node.children.items():
            result["children"][token_id] = self._traverse_for_sharing(child_node)

        return result

    def _path_to_string(self, token_ids: List[int]) -> str:
        """Convert token path to string key"""
        return "_".join(map(str, token_ids))

    def get_prefix_groups(self) -> Dict[str, List[str]]:
        """
        Get groups of requests that share common prefixes

        Returns:
            Dict mapping prefix signatures to lists of request IDs
        """
        # Implement prefix group identification for SGLang scheduling:
        # 1. Analyze the tree structure
        # 2. Group requests by shared prefixes
        # 3. Provide groupings for scheduler optimization
        groups = {}
        self._collect_groups(self.root, groups, [])

        # Filter out groups with only one request (no sharing possible)
        filtered_groups = {prefix: req_list for prefix, req_list in groups.items()
                          if len(req_list) > 1}

        # Sort requests in each group for consistent ordering
        for prefix in filtered_groups:
            filtered_groups[prefix].sort()

        return filtered_groups

    def _collect_groups(
        self, node: RadixTreeNode, groups: Dict, current_path: List[int]
    ):
        """Helper to collect prefix groups"""
        if len(node.request_ids) > 1:  # Multiple requests share this prefix
            path_str = self._path_to_string(current_path)
            groups[path_str] = node.request_ids[:]

        for token_id, child_node in node.children.items():
            self._collect_groups(child_node, groups, current_path + [token_id])


class RequestPrefixMatcher:
    """
    SGLang-style prefix matching system for grouping requests with computation sharing
    """

    def __init__(self):
        self.radix_tree = RadixTree()
        self.max_prefix_cache_size = 10000  # Limit for efficiency

    def add_request_prefix(self, request_id: str, prompt_tokens: List[int]):
        """
        Add a request's prefix to the matching system

        Args:
            request_id: ID of the request
            prompt_tokens: Tokenized prompt
        """
        # Implement SGLang-style prefix addition:
        # 1. Add to radix tree
        # 2. Update prefix similarity measures
        # 3. Prepare for computation sharing
        self.radix_tree.insert_request(request_id, prompt_tokens)

        # Update prefix similarity measures
        self._update_prefix_similarity(request_id, prompt_tokens)

    def _update_prefix_similarity(self, request_id: str, prompt_tokens: List[int]):
        """Update similarity measures for the given request and tokens."""
        # For now, we'll calculate a simple similarity measure based on shared prefixes
        # In a full implementation, this would use more sophisticated algorithms
        shared_requests, _ = self.radix_tree.find_shared_prefixes(prompt_tokens)

        # Store similarity information for optimization
        # This is a simplified version - in practice, you'd want to store more detailed
        # similarity metrics
        for shared_request in shared_requests:
            if shared_request != request_id:
                # Process similarity between this and shared_request
                # For now, just ensure they're in the same prefix group
                pass

    def find_computation_sharing_opportunities(
        self, new_request_id: str, new_tokens: List[int]
    ) -> List[str]:
        """
        Find existing requests that can share computation with a new request

        Args:
            new_request_id: ID of new request
            new_tokens: Tokens of new request

        Returns:
            List of request IDs that can share computation
        """
        # Implement SGLang-style computation sharing identification:
        # 1. Find requests with matching prefixes
        # 2. Calculate sharing efficiency
        # 3. Return optimal sharing groups
        shared_requests, prefix_length = self.radix_tree.find_shared_prefixes(new_tokens)

        # Calculate sharing efficiency and filter based on effectiveness
        if prefix_length > 0:  # Only consider sharing if there's a meaningful prefix match
            efficiency_requests = []
            for request_id in shared_requests:
                # We could implement more complex efficiency calculations here
                # For now, we'll just include all shared requests
                efficiency_requests.append(request_id)

            return efficiency_requests
        else:
            return []

    def remove_request(self, request_id: str):
        """Remove a request from the prefix matching system"""
        # Implement removal from prefix system
        self.radix_tree.remove_request(request_id)

        # Any additional cleanup for the prefix matching system
        # could be done here if needed

    def get_optimization_suggestions(self) -> Dict[str, Any]:
        """
        Get suggestions for computation optimization based on prefix analysis

        Returns:
            Dict with optimization recommendations
        """
        # Implement SGLang-style optimization analysis:
        # 1. Analyze sharing opportunities
        # 2. Estimate performance gains
        # 3. Suggest scheduling optimizations
        sharing_graph = self.radix_tree.get_shared_computation_graph()
        prefix_groups = self.radix_tree.get_prefix_groups()

        # Identify the best sharing opportunities
        sharing_opportunities = sharing_graph.get("sharing_opportunities", [])

        # Calculate more detailed performance estimates
        detailed_performance = self._analyze_performance_gains(sharing_opportunities, prefix_groups)

        return {
            "sharing_graph": sharing_graph,
            "prefix_groups": prefix_groups,
            "total_sharing_opportunities": len(sharing_opportunities),
            "estimated_speedup_factor": self._estimate_speedup(prefix_groups),
            "performance_analysis": detailed_performance,
            "top_sharing_opportunities": self._get_top_sharing_opportunities(sharing_opportunities),
            "recommended_batch_sizes": self._recommend_batch_sizes(prefix_groups),
        }

    def _analyze_performance_gains(self, sharing_opportunities, prefix_groups):
        """Analyze potential performance gains from sharing."""
        if not sharing_opportunities:
            return {
                "estimated_tokens_saved": 0,
                "max_concurrent_sharing": 0,
                "potential_latency_reduction": 0.0,
                "memory_savings_estimate": 0,
            }

        total_tokens_saved = sum(opp.get("potential_savings", 0) * len(opp.get("path", []))
                                for opp in sharing_opportunities)
        max_concurrent_sharing = max((len(opp.get("requests", [])) for opp in sharing_opportunities), default=0)

        # Estimate latency reduction (simplified)
        latency_reduction = min(0.5, max_concurrent_sharing * 0.1)  # Up to 50% reduction

        # Estimate memory savings from shared KV caches
        memory_savings = len(prefix_groups)  # Simplified estimation

        return {
            "estimated_tokens_saved": total_tokens_saved,
            "max_concurrent_sharing": max_concurrent_sharing,
            "potential_latency_reduction": latency_reduction,
            "memory_savings_estimate": memory_savings,
        }

    def _get_top_sharing_opportunities(self, sharing_opportunities):
        """Get the top sharing opportunities sorted by benefit."""
        # Sort by potential savings (highest first)
        sorted_opportunities = sorted(
            sharing_opportunities,
            key=lambda x: x.get("potential_savings", 0),
            reverse=True
        )
        return sorted_opportunities[:5]  # Return top 5

    def _recommend_batch_sizes(self, prefix_groups):
        """Recommend batch sizes based on prefix group sizes."""
        group_sizes = [len(requests) for requests in prefix_groups.values()]
        if not group_sizes:
            return {"recommended_batch_sizes": [1, 2, 4]}

        # Return common batch sizes that work well with the group distribution
        unique_sizes = list(set(group_sizes))
        recommended = sorted(unique_sizes + [1, 2, 4, 8])  # Include standard sizes
        return {"recommended_batch_sizes": recommended}

    def _estimate_speedup(self, prefix_groups: Dict[str, List[str]]) -> float:
        """Estimate potential speedup from prefix sharing"""
        # Simple estimation: if we have groups of shared computation,
        # we can process shared parts once instead of multiple times
        total_requests = sum(len(requests) for requests in prefix_groups.values())
        shared_parts = len(prefix_groups)

        if total_requests > 0:
            return total_requests / max(shared_parts, 1)  # Potential speedup
        return 1.0
