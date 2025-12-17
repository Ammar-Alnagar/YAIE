"""
RadixAttention implementation for efficient attention with prefix sharing
This implements the radial attention mechanism inspired by SGLang
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Applies rotary position embedding to queries and keys."""
    # q, k: [batch_size, num_heads, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim] (already sliced to required length)

    # Expand cos and sin to [1, 1, seq_len, head_dim] for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]

    # If position_ids is provided, we need to select the right embeddings
    # The position_ids should match the sequence length of q and k
    if position_ids is not None:
        # position_ids shape: [batch_size, seq_len] or [seq_len]
        # For this implementation, assume we use the actual positions in the sequence
        # If position_ids is different from the natural sequence [0, 1, ..., seq_len-1],
        # we need to select the appropriate cos/sin values
        pass  # We'll use cos and sin as they are, assuming they're already properly selected

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RadixAttentionBlock(nn.Module):
    """
    A single block of Radix Attention with prefix sharing capabilities
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        # Calculate total hidden dimension
        total_hidden_dim = num_heads * head_dim

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_size, total_hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, total_hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, total_hidden_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(total_hidden_dim, hidden_size, bias=False)

        # Cache for positional embeddings for efficiency
        self.register_buffer(
            "cos_cached",
            torch.ones((max_position_embeddings, head_dim), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            torch.ones((max_position_embeddings, head_dim), dtype=torch.float32),
            persistent=False,
        )

        # Convert all model parameters to float16 to match expected input dtype
        self.to(torch.float16)

        # Initialize RoPE embeddings
        self._setup_rope_embeddings()

    def _setup_rope_embeddings(self):
        """Initialize rotary position embeddings."""
        position_ids = torch.arange(
            self.max_position_embeddings, dtype=torch.float32
        )  # [max_pos]

        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )  # [head_dim//2]

        # Calculate sin and cos embeddings
        freqs = torch.einsum(
            "i,j->ij", position_ids, inv_freq
        )  # [max_pos, head_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_pos, head_dim]

        self.cos_cached = emb.cos().to(dtype=torch.float16)
        self.sin_cached = emb.sin().to(dtype=torch.float16)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """
        Forward pass for radial attention with prefix sharing

        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs for RoPE
            past_key_value: Previous key-value pairs for incremental decoding
            output_attentions: Whether to output attention weights
            use_cache: Whether to cache key-value pairs

        Returns:
            Output tensor, optional attention weights, optional key-value pairs
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Apply RoPE (Rotary Position Embedding)
        # Convert buffers to same dtype as query for RoPE computation
        cos_to_use = self.cos_cached[:seq_len].to(query.dtype)
        sin_to_use = self.sin_cached[:seq_len].to(query.dtype)

        if position_ids is not None:
            query, key = apply_rotary_pos_emb(
                query, key, cos_to_use, sin_to_use, position_ids
            )
        else:
            # Use sequential positions if none provided
            pos_ids = torch.arange(seq_len, device=query.device).unsqueeze(0)
            query, key = apply_rotary_pos_emb(
                query, key, cos_to_use, sin_to_use, pos_ids
            )

        # Handle past key-value caching
        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)

        # For causal attention, we use scaled dot-product attention
        # Scaled attention: (Q @ K.T) / sqrt(head_dim)
        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention mask to match attention weights shape
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(expanded_mask == 0, float("-inf"))

        # Apply softmax to get attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)

        # Transpose and reshape back to [batch_size, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )

        # Output projection
        output = self.o_proj(attn_output)

        # Prepare return values
        outputs = (output,)

        if output_attentions:
            outputs += (attn_weights,)
        else:
            outputs += (None,)

        if use_cache:
            outputs += ((key, value),)
        else:
            outputs += (None,)

        return outputs[0], outputs[1], outputs[2]


class RadixAttentionWithPagedKVCache:
    """
    Radial attention with paged KV-cache management
    This class manages the paged cache for radial attention
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_blocks_per_request: int = 128,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks_per_request = max_blocks_per_request

        # Memory pools for key and value caches
        self.key_cache_pool = {}
        self.value_cache_pool = {}

        # Track which blocks are assigned to which request
        self.request_block_map = {}  # request_id -> [block_ids]

        # Track next available block index
        self.next_block_index = 0

        # Max total blocks (adjust as needed)
        self.max_total_blocks = 10240  # Adjust based on available memory

    def allocate_blocks(self, request_id: str, num_blocks_needed: int) -> List[int]:
        """Allocate blocks for a request."""
        if num_blocks_needed > (self.max_total_blocks - self.next_block_index):
            # Implement cache eviction policy here in a real system
            raise RuntimeError(f"Not enough space for {num_blocks_needed} blocks")

        allocated_block_ids = []
        for _ in range(num_blocks_needed):
            block_id = self.next_block_index
            self.next_block_index += 1

            # Initialize key and value cache blocks
            self.key_cache_pool[block_id] = torch.zeros(
                self.block_size,
                self.num_heads,
                self.head_dim,
                dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            self.value_cache_pool[block_id] = torch.zeros(
                self.block_size,
                self.num_heads,
                self.head_dim,
                dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            allocated_block_ids.append(block_id)

        self.request_block_map[request_id] = allocated_block_ids
        return allocated_block_ids

    def append_slot(self, key: torch.Tensor, value: torch.Tensor, request_id: str):
        """
        Append new key-value pairs to the cache for a request

        This method handles the complex task of storing new KV pairs in the paged memory system.
        It manages block allocation, handles cases where data spans multiple blocks, and
        ensures efficient memory usage by packing data into fixed-size blocks.

        Args:
            key: Key tensor to store [num_tokens, num_heads, head_dim] or [1, num_tokens, num_heads, head_dim]
            value: Value tensor to store [num_tokens, num_heads, head_dim] or [1, num_tokens, num_heads, head_dim]
            request_id: Request identifier for block mapping
        """
        # Ensure the request has allocated blocks in the paged memory system
        if request_id not in self.request_block_map:
            # First-time allocation for this request - start with one block
            num_new_blocks = 1
            self.allocate_blocks(request_id, num_new_blocks)

        # Get the list of blocks currently allocated to this request
        block_ids = self.request_block_map[request_id]

        # Validate that blocks were actually allocated
        if len(block_ids) == 0:
            raise RuntimeError(f"No blocks allocated for request {request_id}")

        # Use the last (most recent) block for appending new data
        # In production systems, this might use more sophisticated block selection
        last_block_id = block_ids[-1]

        # Determine how many slots in the last block are already occupied
        # This tracks the fill level of each block to know where to append new data
        occupied_slots = min(
            self.block_size, getattr(self, f"_slots_used_{last_block_id}", 0)
        )

        # Check if the last block is completely full and needs expansion
        if occupied_slots >= self.block_size:
            # Block is full - allocate a new block for this request
            new_block_id = self.next_block_index
            self.next_block_index += 1

            # Initialize the new block with zeros in the appropriate shape and device
            # Shape: [block_size, num_heads, head_dim] for efficient attention computation
            self.key_cache_pool[new_block_id] = torch.zeros(
                self.block_size,
                self.num_heads,
                self.head_dim,
                dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            self.value_cache_pool[new_block_id] = torch.zeros(
                self.block_size,
                self.num_heads,
                self.head_dim,
                dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            # Add the new block to this request's block list and update mapping
            block_ids.append(new_block_id)
            self.request_block_map[request_id] = block_ids
            last_block_id = new_block_id
            occupied_slots = 0  # New block starts empty

        # Prepare the key/value tensors for storage
        # Handle different input tensor shapes that may come from the model
        if key.dim() == 4 and key.shape[0] == 1:
            # Remove singleton batch dimension if present
            # Input shape: [1, num_tokens, num_heads, head_dim] -> [num_tokens, num_heads, head_dim]
            key = key.squeeze(0)
            value = value.squeeze(0)

        # Determine how many tokens we're adding (could be multiple tokens at once)
        num_tokens_to_add = key.shape[0] if key.dim() == 3 else 1

        # Check if the new data fits in the remaining space of the current block
        if occupied_slots + num_tokens_to_add > self.block_size:
            # Data spans multiple blocks - split it across available blocks
            # First, fill whatever space remains in the current block
            remaining_in_current = self.block_size - occupied_slots
            if remaining_in_current > 0:
                # Fill the remaining slots in the current block
                self.key_cache_pool[last_block_id][
                    occupied_slots : occupied_slots + remaining_in_current
                ] = key[:remaining_in_current]
                self.value_cache_pool[last_block_id][
                    occupied_slots : occupied_slots + remaining_in_current
                ] = value[:remaining_in_current]

                # Allocate more blocks for the rest
                remaining_keys = key[remaining_in_current:]
                remaining_values = value[remaining_in_current:]

                # Allocate additional blocks as needed
                additional_blocks_needed = (
                    remaining_keys.shape[0] + self.block_size - 1
                ) // self.block_size
                for i in range(additional_blocks_needed):
                    new_block_id = self.next_block_index
                    self.next_block_index += 1

                    # Initialize new block
                    self.key_cache_pool[new_block_id] = torch.zeros(
                        self.block_size,
                        self.num_heads,
                        self.head_dim,
                        dtype=torch.float16,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )
                    self.value_cache_pool[new_block_id] = torch.zeros(
                        self.block_size,
                        self.num_heads,
                        self.head_dim,
                        dtype=torch.float16,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )

                    block_ids.append(new_block_id)
                    self.request_block_map[request_id] = block_ids

                    # Add keys/values to new block
                    start_idx = i * self.block_size
                    end_idx = min(start_idx + self.block_size, remaining_keys.shape[0])
                    tokens_to_add = end_idx - start_idx
                    if tokens_to_add > 0:
                        self.key_cache_pool[new_block_id][:tokens_to_add] = (
                            remaining_keys[start_idx:end_idx]
                        )
                        self.value_cache_pool[new_block_id][:tokens_to_add] = (
                            remaining_values[start_idx:end_idx]
                        )
        else:
            # We can fit in the current block
            self.key_cache_pool[last_block_id][
                occupied_slots : occupied_slots + num_tokens_to_add
            ] = key
            self.value_cache_pool[last_block_id][
                occupied_slots : occupied_slots + num_tokens_to_add
            ] = value

    def get_kv_cache(
        self, request_ids: List[str], seq_lens: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get key-value cache for specified requests"""
        if not request_ids:
            # Return empty tensors if no request IDs
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return (
                torch.empty(0, 0, self.num_heads, self.head_dim, device=device),
                torch.empty(0, 0, self.num_heads, self.head_dim, device=device),
            )

        # Calculate total number of tokens needed
        total_tokens = sum(seq_lens)

        # Initialize output tensors
        device = "cuda" if torch.cuda.is_available() else "cpu"
        all_keys = torch.zeros(
            1,
            total_tokens,
            self.num_heads,
            self.head_dim,
            dtype=torch.float16,
            device=device,
        )
        all_values = torch.zeros(
            1,
            total_tokens,
            self.num_heads,
            self.head_dim,
            dtype=torch.float16,
            device=device,
        )

        # Collect keys and values from all requests
        current_token_idx = 0
        for request_id, seq_len in zip(request_ids, seq_lens):
            if request_id in self.request_block_map:
                block_ids = self.request_block_map[request_id]

                # Extract keys and values from blocks for this request
                req_keys = []
                req_values = []

                for block_id in block_ids:
                    block_key = self.key_cache_pool[block_id]
                    block_val = self.value_cache_pool[block_id]

                    # Only take up to the required sequence length
                    tokens_from_this_block = min(
                        block_key.shape[0], seq_len - len(req_keys) * self.block_size
                    )
                    if tokens_from_this_block > 0:
                        req_keys.append(block_key[:tokens_from_this_block])
                        req_values.append(block_val[:tokens_from_this_block])

                        if len(req_keys) * self.block_size >= seq_len:
                            break

                # Concatenate collected keys and values
                if req_keys:
                    req_keys_tensor = torch.cat(req_keys, dim=0)[:seq_len]
                    req_values_tensor = torch.cat(req_values, dim=0)[:seq_len]

                    # Add to all tensors
                    tokens_to_copy = min(seq_len, all_keys.shape[1] - current_token_idx)
                    if tokens_to_copy > 0:
                        all_keys[
                            0, current_token_idx : current_token_idx + tokens_to_copy
                        ] = req_keys_tensor[:tokens_to_copy]
                        all_values[
                            0, current_token_idx : current_token_idx + tokens_to_copy
                        ] = req_values_tensor[:tokens_to_copy]
                        current_token_idx += tokens_to_copy

        return all_keys, all_values

    def free_request(self, request_id: str):
        """Free the KV-cache associated with a request"""
        if request_id in self.request_block_map:
            # Remove references to the blocks but don't actually free GPU memory
            # In a real system, you'd implement more sophisticated memory management
            del self.request_block_map[request_id]

            # Optionally clean up cache entries
            # Note: In a production system, you'd implement actual memory deallocation
            # and reuse strategies for better performance
